#!/usr/bin/env python3
"""Resident launcher for one core slot. Run it in that slot's screen.

Usage: slotd.py <slot> [--no-screen-check]

The slot's mode file, <state_dir>/slot<N>.mode, says what slotd does:

runplan (the default): wait for a trigger written by runlaunch.py, then run
runplan<slot>.sh from the cases directory.

queue: start the jobs runq.py queued for this slot, oldest first, together
with the `any` jobs this slot may take. Each job's checks run again at start;
a refused job ends as `refused`, with its reasons. A job whose --after job did
not end ok ends as `skipped`. `runq cancel` or Ctrl-C stops the running job
and holds the slot until `runq go`; a job past its --timeout is stopped too.
Jobs left running when a slotd stopped end as `lost`.

Every run is pinned to the slot's cores with taskset, holds the slot's run
lock and gets HERMES_SLOT=<slot>; a queued job also gets RUNQ_JOB=<id>. Each
run loses the variables that mark an agent's shell (CLAUDE*, CODEX*,
AI_AGENT*) and SDTOOLS_SLOT_ENFORCE. sdrun.py refuses to start without
HERMES_SLOT once the slot config enforces it, so a run can only start
through this loop.

slotd restarts itself, in place, when it is idle and its mode file has
changed, or a new commit has changed its source (slotd.py, slotlib.py,
runcheck.py). An uncommitted edit to the source waits for its commit.

State, heartbeat and last exit code go to <state_dir>/slot<N>.state, which
runlaunch.py, runwait.py, runq.py and runstat.py read. Ctrl-C while idle
stops slotd.
"""
import datetime as dt
import fcntl
import glob
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import runcheck  # noqa: E402
import slotlib  # noqa: E402

POLL_S = 1
TRIGGER_MAX_AGE_S = 15   # runlaunch.py waits this long; an older trigger is stale
KILL_AFTER_S = 60        # SIGTERM, then SIGKILL this long after
SOURCES = ("slotd.py", "slotlib.py", "runcheck.py")
OUTPUT = ("BOUT.log.*", "BOUT.dmp.*", "BOUT.restart.*", "BOUT.settings")


def log(slot, msg):
    print(f"[slotd {slot} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def run_env(slot, job_id=None):
    """slotd's environment for a run: without the agent markers or the
    enforcement override, with the slot and, for a queued job, its id."""
    env = {k: v for k, v in os.environ.items()
           if not runcheck.AGENT_MARK.match(k) and
           k not in ("SDTOOLS_SLOT_ENFORCE", "RUNQ_JOB")}
    env["HERMES_SLOT"] = str(slot)
    if job_id:
        env["RUNQ_JOB"] = job_id
    return env


def stamp(path):
    try:
        s = os.stat(path)
        return (path, s.st_mtime_ns, s.st_size)
    except FileNotFoundError:
        return (path, None, None)


def code_version():
    """The last commit that changed slotd's sources, or None while one of
    them differs from it: an edit nobody has committed yet. Outside a git
    checkout, the sources' times and sizes."""
    try:
        git = [subprocess.run(["git", "-C", HERE, *cmd, "--", *SOURCES],
                              capture_output=True, text=True, timeout=10)
               for cmd in (["log", "-1", "--format=%H"],
                           ["status", "--porcelain"])]
    except (OSError, subprocess.SubprocessError):
        git = None
    if git is None or any(g.returncode for g in git) or \
            not git[0].stdout.strip():
        return [stamp(os.path.join(HERE, n)) for n in SOURCES]
    return None if git[1].stdout.strip() else git[0].stdout.strip()


def fingerprint(cfg, slot, seen=None):
    """What a change of restarts slotd: its committed sources and its mode
    file. An uncommitted edit counts only once it is committed, so editing
    sdtools in place never puts untested code into a live slot."""
    code = code_version()
    if code is None:
        code = seen[0] if seen else None
    return (code, stamp(slotlib.path(cfg, slot, "mode")))


def take_lock(cfg, slot):
    """The slot's run lock, held, or None if another process holds it."""
    lock = open(slotlib.path(cfg, slot, "lock"), "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return None
    return lock


# --- runplan mode --------------------------------------------------------

def run_once(cfg, slot, trigger, st):
    plan = slotlib.runplan(cfg, slot)
    lock = take_lock(cfg, slot)
    if lock is None:
        log(slot, "run lock is held by another process; launch refused")
        st.update(last_error="run lock held", last_launch=trigger.get("id"))
        return
    cores = cfg["cores"][str(slot)]
    log(slot, f"launch {trigger.get('id')}: {plan} on cores {cores}")
    # The runplan inherits the lock, so the slot stays locked while it or
    # anything it started in the background lives, even if slotd dies.
    child = subprocess.Popen(["taskset", "-c", cores, "bash", plan],
                             cwd=cfg["cases"], env=run_env(slot),
                             pass_fds=(lock.fileno(),))
    st.update(state="running", pid=child.pid, since=time.time(),
              launch=trigger.get("id"), last_error=None)
    slotlib.write_state(cfg, slot, st)
    try:
        while child.poll() is None:
            time.sleep(POLL_S)
            slotlib.write_state(cfg, slot, st)
    except KeyboardInterrupt:
        log(slot, "Ctrl-C: waiting for the runplan to stop")
        while child.poll() is None:
            try:
                child.wait()
            except KeyboardInterrupt:
                log(slot, "second Ctrl-C: terminating the runplan")
                child.terminate()
    finally:
        # Closing, not unlocking: LOCK_UN would release the lock the runplan's
        # background children still share.
        lock.close()
    log(slot, f"runplan ended, exit {child.returncode}")
    st.update(state="idle", pid=None, last_exit=child.returncode,
              last_end=time.time())
    slotlib.write_state(cfg, slot, st)


def runplan_tick(cfg, slot, st, go):
    orphan = not slotlib.lock_free(cfg, slot)
    st["state"] = "orphan" if orphan else "idle"
    if not os.path.exists(go):
        return
    try:
        with open(go, encoding="utf-8") as f:
            trigger = json.load(f)
    except (OSError, ValueError):
        trigger = {"id": "unreadable"}
    os.remove(go)
    if time.time() - trigger.get("time", 0) > TRIGGER_MAX_AGE_S:
        log(slot, f"ignored stale trigger {trigger.get('id')}")
    elif orphan:
        log(slot, "an earlier runplan still holds the run lock; launch "
            "refused")
        st.update(last_error="earlier runplan still running",
                  last_launch=trigger.get("id"))
    else:
        run_once(cfg, slot, trigger, st)
        log(slot, f"waiting for runlaunch.py {slot}")


# --- queue mode ----------------------------------------------------------

def job_cores(job):
    """The cores a job names with -c=, as a set, else None."""
    spec = next((w[3:] for w in job.get("argv") or [] if w.startswith("-c=")),
                None)
    return slotlib.cpu_set(spec) if spec else None


def may_take(job, owner, slot):
    """True if this slot, with this owner, may take this `any` job: the
    owner's campaign's job, or a user's job outside every campaign. A job
    added under a mandate runs only on the mandate's slots."""
    if job.get("only_slots") and str(slot) not in job["only_slots"]:
        return False
    if job.get("campaign") is None:
        return (job.get("by") or {}).get("kind") == "user"
    return job["campaign"] == owner


def move_output(case):
    """Move a case's output into <case>/overwritten-<time>/, for --rerun."""
    dest = os.path.join(case, "overwritten-" +
                        dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    files = sorted({p for pat in OUTPUT
                    for p in glob.glob(os.path.join(case, pat))})
    if files:
        os.makedirs(dest)
        for p in files:
            os.rename(p, os.path.join(dest, os.path.basename(p)))
    return dest if files else None


def validate_copy(cfg, job):
    """Copy the case's input files, without its output or subdirectories,
    to validate/<id>/<case name>/; return the copy's path."""
    case = job["case"]
    dest = os.path.join(cfg["state_dir"], "validate", job["id"],
                        os.path.basename(case))
    os.makedirs(dest)
    skip = {p for pat in OUTPUT + ("validate.*",)
            for p in glob.glob(os.path.join(case, pat))}
    for name in os.listdir(case):
        p = os.path.join(case, name)
        if os.path.isfile(p) and p not in skip:
            shutil.copy2(p, dest)
    return dest


class Queue:
    """slotd's queue-mode state: the jobs it runs, and the lock they share."""

    def __init__(self, cfg, slot, st):
        self.cfg, self.slot, self.st = cfg, str(slot), st
        self.cores = cfg["cores"][self.slot]
        self.lock = None
        self.runs = []      # dicts: job, proc, deadline, stop_at, outcome

    # The store -------------------------------------------------------------

    def save(self, job, where="running"):
        slotlib.write_job(self.cfg, where, self.slot, job)

    def end(self, job, outcome, why=()):
        """Write a job's outcome and move it to done/."""
        job.update(outcome=outcome, ended=job.get("ended") or now(),
                   slot_ran=self.slot)
        if why:
            job["why"] = list(why)
        self.save(job)
        slotlib.move_job(self.cfg, job["id"], ("running", self.slot),
                         ("done", self.slot))
        log(self.slot, f"job {job['id']} {outcome}"
            + (f": {'; '.join(job['why'])}" if job.get("why") else ""))

    def recover(self):
        """End every job a previous slotd left in running/ as lost."""
        for job in slotlib.read_jobs(self.cfg, "running", self.slot):
            self.end(job, "lost", ["slotd stopped while it ran"])

    def candidates(self):
        """Queued jobs this slot may take, oldest first."""
        own = slotlib.read_jobs(self.cfg, "queue", self.slot)
        try:
            owner = slotlib.owner(self.cfg, self.slot)
        except ValueError:
            owner = None
        anys = [j for j in slotlib.read_jobs(self.cfg, "queue", "any")
                if may_take(j, owner, self.slot)]
        return sorted(own + anys, key=lambda j: j["id"])

    # Starting --------------------------------------------------------------

    def next_job(self):
        """The oldest job, when it may start now, else None. slotd never
        skips past the oldest job."""
        jobs = self.candidates()
        if not jobs:
            return None
        job = jobs[0]
        if job.get("after"):
            where, _ = runcheck.find_job(self.cfg, job["after"])
            if where in ("queue", "running"):
                return None
        if self.runs:
            mine = job_cores(job)
            if mine is None or any(job_cores(r["job"]) is None or
                                   job_cores(r["job"]) & mine
                                   for r in self.runs):
                return None
        return job

    def claim(self, job):
        """Move the job into running/<slot>/; None if another slotd won."""
        src = ("queue", job["slot"])
        p = slotlib.move_job(self.cfg, job["id"], src, ("running", self.slot))
        if p is None:
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def command(self, job):
        """The argv that runs the job, and its console file."""
        if job.get("cmd") is not None:
            out = slotlib.job_path(self.cfg, "done", self.slot,
                                   job["id"])[:-5] + ".log"
            line = shlex.join(job["cmd"])
        else:
            case, argv = job["case"], list(job.get("argv") or [])
            if job.get("validate"):
                case = job["validate_copy"]
                argv = [w for w in argv if w not in ("-restart", "-append")]
                argv.append("input:validate=true")
            pin = [] if job_cores(job) else [f"-s={self.slot}"]
            out = os.path.join(case, "BOUT.log.console")
            line = shlex.join([os.path.join(HERE, "sdrun.py"), *pin, "-y",
                               "-d", case, *argv])
        # A continuation appends to the console, which alone keeps PETSc's
        # finalize output of the earlier runs.
        append = job.get("cmd") is None and not job.get("validate") and \
            any(w in ("-append", "append") for w in job.get("argv") or [])
        line += f" 2>&1 | tee {'-a ' if append else ''}" + shlex.quote(out)
        return ["taskset", "-c", self.cores, "bash", "-o", "pipefail", "-c",
                line], out

    def start(self, job):
        found = runcheck.problems(self.cfg, job, "start", self.slot)
        hard = [r for r in found if not r.warn]
        job.update(slot_ran=self.slot, cores=self.cores)
        if hard:
            skipped = all(r.rule == "after-failed" for r in hard)
            self.end(job, "skipped" if skipped else "refused",
                     [r.message for r in hard])
            return
        try:
            if job.get("rerun"):
                job["moved_to"] = move_output(job["case"])
            if job.get("validate"):
                job["validate_copy"] = validate_copy(self.cfg, job)
            argv, out = self.command(job)
        except OSError as e:
            self.end(job, "failed", [f"could not prepare the job: {e}"])
            return
        if self.lock is None:
            self.lock = take_lock(self.cfg, self.slot)
            if self.lock is None:
                # Another process took the lock since the last look: put
                # the job back and wait.
                del job["slot_ran"], job["cores"]
                self.save(job)
                slotlib.move_job(self.cfg, job["id"], ("running", self.slot),
                                 ("queue", job["slot"]))
                return
        try:
            # start_new_session: the job gets its own process group, so one
            # signal stops it and all it started; Ctrl-C in the screen
            # reaches slotd only.
            proc = subprocess.Popen(argv, cwd=job["cwd"],
                                    env=run_env(self.slot, job["id"]),
                                    pass_fds=(self.lock.fileno(),),
                                    start_new_session=True)
        except OSError as e:
            self.end(job, "failed", [f"could not start: {e}"])
            self.release()
            return
        job.update(started=now(), pid=proc.pid, output=out)
        self.save(job)
        deadline = time.time() + job["timeout_s"] if job.get("timeout_s") \
            else None
        self.runs.append({"job": job, "proc": proc, "t0": time.time(),
                          "deadline": deadline, "stop_at": None,
                          "outcome": None})
        log(self.slot, f"start {job['id']}: {runcheck.line(job)[5:]}")

    def release(self):
        if not self.runs and self.lock is not None:
            # Closing, not unlocking: an orphan the job left keeps the lock.
            self.lock.close()
            self.lock = None

    # Stopping --------------------------------------------------------------

    def stop(self, run, outcome):
        if run["stop_at"] is None:
            run.update(stop_at=time.time(), outcome=outcome)
            log(self.slot, f"stopping {run['job']['id']} ({outcome})")
            self.signal(run, signal.SIGTERM)

    def signal(self, run, sig):
        try:
            os.killpg(run["proc"].pid, sig)
        except ProcessLookupError:
            pass

    def hold(self, why, who=None):
        p = slotlib.path(self.cfg, self.slot, "hold")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"by": who or {"kind": "slotd"}, "when": now(),
                       "why": why}, f)
        os.replace(p + ".tmp", p)

    def cancel_all(self, why, who=None):
        for run in self.runs:
            self.stop(run, "cancelled")
        self.hold(why, who)

    def watch(self):
        """Handle a cancel request, timeouts and ended jobs."""
        cancel = slotlib.path(self.cfg, self.slot, "cancel")
        if os.path.exists(cancel):
            try:
                with open(cancel, encoding="utf-8") as f:
                    req = json.load(f)
            except (OSError, ValueError):
                req = {}
            os.remove(cancel)
            if self.runs:
                self.cancel_all("runq cancel", req.get("by"))
        t = time.time()
        for run in list(self.runs):
            rc = run["proc"].poll()
            if rc is None:
                if run["deadline"] and t > run["deadline"]:
                    self.stop(run, "timeout")
                elif run["stop_at"] and t - run["stop_at"] > KILL_AFTER_S:
                    self.signal(run, signal.SIGKILL)
                continue
            # The group may outlive its leader; stop what is left.
            self.signal(run, signal.SIGKILL if run["stop_at"] else
                        signal.SIGTERM)
            job = run["job"]
            job.update(exit=rc, ended=now(), wall_s=round(t - run["t0"]))
            if job.get("validate"):
                for src, dst in (("BOUT.log.0", "validate.log.0"),
                                 ("BOUT.log.console", "validate.console")):
                    p = os.path.join(job["validate_copy"], src)
                    if os.path.isfile(p):
                        shutil.copy2(p, os.path.join(job["case"], dst))
            self.runs.remove(run)
            self.end(job, run["outcome"] or ("ok" if rc == 0 else "failed"),
                     [f"exit {rc}"] if rc and not run["outcome"] else ())
        self.release()

    def tick(self):
        self.watch()
        held = os.path.exists(slotlib.path(self.cfg, self.slot, "hold"))
        if not self.runs and not held and not slotlib.lock_free(self.cfg,
                                                                self.slot):
            self.st["state"] = "orphan"
            return
        if not held:
            job = self.next_job()
            if job is not None:
                job = self.claim(job)
                if job is not None:
                    self.start(job)
        self.st.update(state="running" if self.runs else
                       ("held" if held else "idle"),
                       jobs=[r["job"]["id"] for r in self.runs],
                       pid=self.runs[0]["proc"].pid if self.runs else None)


# --- main ----------------------------------------------------------------

def main():
    args = sys.argv[1:]
    check_screen = "--no-screen-check" not in args
    args = [a for a in args if a != "--no-screen-check"]
    if len(args) != 1:
        sys.exit(__doc__)
    slot = args[0]
    cfg = slotlib.config()
    if slot not in cfg["cores"]:
        sys.exit(f"unknown slot {slot}; slots are {sorted(cfg['cores'])}")
    if check_screen and not os.environ.get("STY", "").endswith(f".h{slot}"):
        sys.exit(f"slotd {slot} runs only inside screen h{slot} "
                 f"(STY is {os.environ.get('STY') or 'unset'})")
    # Python opens files non-inheritable, so os.execv below closes this
    # lock and the new slotd can take it.
    daemon = open(slotlib.path(cfg, slot, "daemon"), "a")
    try:
        fcntl.flock(daemon, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"slotd {slot} is already running")
    go = slotlib.path(cfg, slot, "go")
    if os.path.exists(go):
        os.remove(go)   # a trigger left from before this slotd started is stale
    seen = fingerprint(cfg, slot)
    try:
        mode, mode_error = slotlib.mode(cfg, slot), None
    except ValueError as e:
        mode, mode_error = None, str(e)
    st = dict(state="idle", slotd_pid=os.getpid(), started=time.time(),
              runplan=slotlib.runplan(cfg, slot), mode=mode,
              mode_error=mode_error)
    queue = Queue(cfg, slot, st) if mode == "queue" else None
    if queue:
        queue.recover()
        log(slot, f"queue mode: waiting for runq add {slot}")
    elif mode == "runplan":
        log(slot, f"runplan mode: waiting for runlaunch.py {slot}")
    else:
        log(slot, f"stopped: {mode_error}")
        st["state"] = "stopped"
    slotlib.write_state(cfg, slot, st)
    while True:
        try:
            busy = queue is not None and bool(queue.runs)
            if not busy and fingerprint(cfg, slot, seen) != seen:
                log(slot, "mode or source changed: restarting")
                st.update(state="restarting")
                slotlib.write_state(cfg, slot, st)
                os.execv(sys.executable, [sys.executable,
                                          os.path.realpath(__file__)]
                         + sys.argv[1:])
            if queue:
                queue.tick()
            elif mode == "runplan":
                runplan_tick(cfg, slot, st, go)
            slotlib.write_state(cfg, slot, st)
            time.sleep(POLL_S)
        except KeyboardInterrupt:
            if queue and queue.runs:
                log(slot, f"Ctrl-C: stopping the job and holding slot {slot}"
                    f" (runq go {slot} resumes)")
                queue.cancel_all("Ctrl-C in screen h" + slot)
                continue
            break
    st.update(state="stopped", pid=None)
    slotlib.write_state(cfg, slot, st)
    log(slot, "stopped")


if __name__ == "__main__":
    main()
