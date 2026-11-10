#!/usr/bin/env python3
"""runq: the run queue. Queue Hermes-3 cases and other jobs on a core slot;
the slot's slotd starts them one at a time, oldest first.

Usage:
    runq add <slot|any> <case> [sdrun.py options] [job options] [--dry-run]
    runq add <slot|any> [job options] -- <command> [args]   [--dry-run]
    runq add <slot|any> -f <batch file> [--dry-run]
    runq ls [--json]
    runq rm <job>
    runq cancel <slot>
    runq hold <slot> [why]
    runq go <slot>
    runq own <slot> <campaign|none> "<the user's words>"
    runq mode <slot> queue|runplan
    runq restart <slot>

Job options: --cwd DIR, --timeout T (e.g. 90s, 30m, 10h), --after JOB,
--study S<nn>, --validate, --rerun. A case job runs
`sdrun.py -s=<slot> -y -d <case> ...` from the case's parent directory unless
--cwd says otherwise. Every case job names its binary with -p or -b. An
agent's job gives --timeout. `any` means the first free slot the job's
campaign owns.

A batch file holds one job per line, as the words after `runq add <slot>`;
blank lines and lines starting with # are skipped. `--after prev` names the
job on the line before. If one line is refused, no line is queued.

--dry-run runs every check and queues nothing.

An agent may add, rm, cancel, hold and go only on a slot its campaign owns,
and rm or cancel only its campaign's jobs. An agent's campaign is the one
whose cases directory or clone holds its working directory. The user may do
everything.

runq restart stops an idle slot's slotd and starts its screen h<slot> again
with the command that started it.

Exit status: 0 queued (or the dry run passed), 1 refused, 2 bad usage.
"""
import datetime as dt
import glob
import json
import os
import shlex
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import runcheck  # noqa: E402
import sdrun  # noqa: E402
import slotlib  # noqa: E402
from runcheck import Refusal  # noqa: E402

RECENT = 5    # ended jobs per slot that `runq ls` shows


class Usage(Exception):
    pass


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def show(refusals, head):
    """Print refusals as `- <message>` with `  fix: <command>` below; return
    the exit status: 1 if any refusal is not a warning, else 0."""
    hard = [r for r in refusals if not r.warn]
    if hard:
        print(head)
    for r in hard + [r for r in refusals if r.warn]:
        print(f"- {'warning: ' if r.warn else ''}{r.message}")
        if r.fix:
            print(f"  fix: {r.fix}")
    return 1 if hard else 0


def slot_arg(cfg, slot, allow_any=False):
    if slot == "any" and allow_any:
        return slot
    if slot not in cfg["cores"]:
        raise Usage(f"slot {slot} does not exist; the slots are "
                    f"{', '.join(sorted(cfg['cores']))}"
                    + (" and any" if allow_any else ""))
    return slot


def label(job):
    return job.get("case") or shlex.join(job.get("cmd") or [])


def by(job):
    who = job.get("by") or {}
    if who.get("kind") == "agent":
        return "agent " + (who.get("tmux") or (who.get("session") or "?")[:8])
    return who.get("kind", "?")


def agent_campaign(cfg, cwd):
    return runcheck.campaign_of(cfg, os.path.realpath(cwd))


store_jobs = slotlib.read_jobs


def owner_or_none(cfg, slot):
    try:
        return slotlib.owner(cfg, slot)
    except ValueError:
        return None


def agent_slot_check(cfg, who, slot, cwd, verb, camp=None):
    """Refusals for an agent acting on a slot its campaign does not own. The
    campaign is `camp` when given (a job's), else the working directory's."""
    if who.get("kind") != "agent":
        return []
    camp = camp or agent_campaign(cfg, cwd)
    if camp is None:
        return [Refusal("agent-campaign", cwd, f"{cwd} lies inside no single "
                        "campaign's cases directory or clone, so runq cannot "
                        f"tell which campaign may {verb} slot {slot}",
                        "cd into your campaign's cases directory or clone")]
    own = owner_or_none(cfg, slot)
    if own != camp:
        return [Refusal("agent-owner", slot, f"slot {slot} belongs to "
                        f"{own or 'no campaign'}, not {camp}",
                        "ask the user for the slot, then record the grant: "
                        f"runq own {slot} {camp} \"<the user's words>\"")]
    return []


def agent_job_check(cfg, who, job, cwd):
    """Refusals for an agent acting on another campaign's job. A job the
    same agent session added always counts as its own."""
    if who.get("kind") != "agent":
        return []
    mine = (job.get("by") or {}).get("session")
    if mine and mine == who.get("session"):
        return []
    camp = agent_campaign(cfg, cwd)
    if camp is None or job.get("campaign") != camp:
        return [Refusal("agent-job", job["id"], f"job {job['id']} belongs to "
                        f"{job.get('campaign') or 'no campaign'}, not "
                        f"{camp or 'your campaign'}")]
    return []


def mode_refusal(cfg, slot, verb):
    try:
        m = slotlib.mode(cfg, slot)
    except ValueError as e:
        return [Refusal("slot-mode", slot, str(e))]
    if m != "queue":
        return [Refusal("slot-mode", slot, f"slot {slot} runs runplans, so "
                        f"there is no queue to {verb}")]
    return []


# --- add ---------------------------------------------------------------

def read_batch(path, cwd):
    try:
        with open(os.path.join(cwd, path), encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise Usage(f"cannot read batch file {path}: {e}")
    lines = []
    for n, s in enumerate(text.splitlines(), 1):
        if s.strip() and not s.lstrip().startswith("#"):
            try:
                lines.append((n, shlex.split(s)))
            except ValueError as e:
                raise Usage(f"{path} line {n}: {e}")
    if not lines:
        raise Usage(f"{path} holds no job")
    return lines


def cmd_add(cfg, args, who, cwd):
    if not args:
        raise Usage("runq add needs a slot")
    slot = slot_arg(cfg, args[0], allow_any=True)
    words = args[1:]
    cut = words.index("--") if "--" in words else len(words)
    head, tail = words[:cut], words[cut:]
    dry = "--dry-run" in head
    head = [w for w in head if w != "--dry-run"]
    if head[:1] == ["-f"]:
        if len(head) != 2 or tail:
            raise Usage("give -f <batch file> and nothing else")
        lines = read_batch(head[1], cwd)
    elif "-f" in head:
        raise Usage("-f comes straight after the slot: runq add <slot> -f "
                    "<batch file>")
    else:
        lines = [(0, head + tail)]

    jobs, prev = [], None
    for n, line_words in lines:
        if "prev" in line_words and n:
            k = line_words.index("prev")
            if k and line_words[k - 1] == "--after" and \
                    "--" not in line_words[:k]:
                if prev is None:
                    raise Usage(f"line {n}: --after prev needs a line before "
                                "it")
                line_words = line_words[:k] + [prev["id"]] + line_words[k + 1:]
        try:
            job = runcheck.make_job(cfg, slot, line_words, cwd=cwd, who=who)
        except ValueError as e:
            raise Usage((f"line {n}: " if n else "") + str(e))
        jobs.append((n, job))
        prev = job

    ids = {j["id"] for _, j in jobs}
    with slotlib.queue_lock(cfg):
        found = []
        for k, (n, job) in enumerate(jobs):
            mine = [r for r in runcheck.problems(cfg, job, "add")
                    if not (r.rule == "after-unknown" and r.target in ids)]
            for _, other in jobs[:k]:
                if job.get("case") and other.get("case") == job["case"] and \
                        job.get("after") != other["id"] and \
                        not job.get("validate") and not other.get("validate"):
                    mine.append(Refusal("duplicate", job["case"],
                                        f"{job['case']} is on an earlier line "
                                        "of the batch too"))
            for r in mine:
                if n:
                    r.message = f"line {n}: {r.message}"
            job["warnings"] = [r.message for r in mine if r.warn]
            found += mine
        what = "dry run" if dry else "nothing queued"
        status = show(found, f"runq add: refused, {what}")
        if status or dry:
            if not status:
                print(f"runq add: dry run passes for {len(jobs)} job"
                      f"{'s' * (len(jobs) != 1)}; nothing queued")
            return status
        for _, job in jobs:
            slotlib.write_job(cfg, "queue", slot, job)
    for _, job in jobs:
        after = f", after {job['after']}" if job.get("after") else ""
        print(f"queued {job['id']} on slot {slot}{after}: {label(job)}")
    return 0


# --- ls ----------------------------------------------------------------

def progress(case):
    import runstat
    try:
        done, nout, _, _ = runstat.case_progress(case)
    except OSError:
        return ""
    return f"{done}/{nout if nout is not None else '?'}"


def snapshot(cfg):
    out = {"slots": {}, "any": store_jobs(cfg, "queue", "any")}
    for n in sorted(cfg["cores"]):
        try:
            m, err = slotlib.mode(cfg, n), None
        except ValueError as e:
            m, err = "bad", str(e)
        hold = None
        try:
            with open(slotlib.path(cfg, n, "hold"), encoding="utf-8") as f:
                hold = json.load(f)
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            hold = {"why": "unreadable hold file"}
        st = slotlib.read_state(cfg, n)
        out["slots"][n] = {
            "mode": m, "mode_error": err, "owner": owner_or_none(cfg, n),
            "cores": cfg["cores"][n], "hold": hold,
            "slotd": st.get("state") if st.get("alive") else "not running",
            "running": store_jobs(cfg, "running", n),
            "queued": store_jobs(cfg, "queue", n),
            "done": store_jobs(cfg, "done", n)[-RECENT:]}
    return out


def cmd_ls(cfg, args, who, cwd):
    if args not in ([], ["--json"]):
        raise Usage("runq ls takes only --json")
    snap = snapshot(cfg)
    if args:
        json.dump(snap, sys.stdout, indent=1)
        print()
        return 0
    for n, s in snap["slots"].items():
        if s["mode"] == "bad":
            print(f"slot {n}  STOPPED  {s['mode_error']}")
            continue
        if s["mode"] == "runplan":
            print(f"slot {n}  {s['owner'] or '—'}  runplan mode, slotd "
                  f"{s['slotd']}")
            continue
        state = "RUNNING" if s["running"] else (
            "HELD" if s["hold"] else "IDLE")
        print(f"slot {n}  {s['owner'] or '—'}  {state}  slotd {s['slotd']}")
        if s["hold"]:
            print(f"        held by {by(s['hold'])}: "
                  f"{s['hold'].get('why') or 'no reason given'}")
        for j in s["running"]:
            p = progress(j["case"]) if j.get("case") else ""
            print(f"        running {j['id']}  {label(j)}  {p}  since "
                  f"{(j.get('started') or '?')[11:16]}  {by(j)}")
        for j in s["queued"]:
            print(f"        queued  {j['id']}  {label(j)}  {by(j)}")
        for j in s["done"]:
            print(f"        ended   {j['id']}  {j.get('outcome', '?')}  "
                  f"{label(j)}")
    for j in snap["any"]:
        print(f"any     queued  {j['id']}  {label(j)}  {by(j)}")
    return 0


# --- rm, cancel, hold, go ----------------------------------------------

def cmd_rm(cfg, args, who, cwd):
    if len(args) != 1:
        raise Usage("runq rm takes one job id")
    with slotlib.queue_lock(cfg):
        where, job = runcheck.find_job(cfg, args[0])
        if where is None:
            return show([Refusal("job-unknown", args[0], f"no job {args[0]} "
                                 "exists", "runq ls")], "runq rm: refused")
        if where != "queue":
            fix = f"runq cancel {job.get('slot_ran') or job['slot']}" \
                if where == "running" else ""
            return show([Refusal("job-not-queued", job["id"],
                                 f"job {job['id']} is {where}, not queued",
                                 fix)], "runq rm: refused")
        found = agent_job_check(cfg, who, job, cwd)
        if not found and job["slot"] != "any":
            found = agent_slot_check(cfg, who, job["slot"], cwd, "use",
                                     job.get("campaign"))
        waiting = [j["id"] for _, j in runcheck.jobs(cfg, ("queue",))
                   if j.get("after") == job["id"]]
        if waiting:
            found.append(Refusal("job-needed", job["id"], f"job "
                                 f"{', '.join(waiting)} waits on job "
                                 f"{job['id']} with --after",
                                 f"runq rm {waiting[0]}"))
        if show(found, "runq rm: refused"):
            return 1
        os.remove(slotlib.job_path(cfg, "queue", job["slot"], job["id"]))
    print(f"removed {job['id']}: {label(job)}")
    return 0


def cmd_cancel(cfg, args, who, cwd):
    if len(args) != 1:
        raise Usage("runq cancel takes one slot")
    slot = slot_arg(cfg, args[0])
    found = mode_refusal(cfg, slot, "cancel")
    jobs = [] if found else store_jobs(cfg, "running", slot)
    if not found and not jobs:
        found.append(Refusal("nothing-running", slot, f"slot {slot} runs no "
                             "job", "runq rm <job> drops a queued job"))
    for j in jobs if not found else []:
        found += agent_job_check(cfg, who, j, cwd) or agent_slot_check(
            cfg, who, slot, cwd, "cancel on", j.get("campaign"))
    if show(found, "runq cancel: refused"):
        return 1
    p = slotlib.path(cfg, slot, "cancel")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"by": who, "when": now(),
                   "jobs": [j["id"] for j in jobs]}, f)
    os.replace(p + ".tmp", p)
    print(f"slot {slot}: cancel requested for {', '.join(j['id'] for j in jobs)}"
          f"; slotd stops it and holds the slot (runq go {slot} resumes)")
    return 0


def cmd_hold(cfg, args, who, cwd):
    if not args:
        raise Usage("runq hold takes a slot and an optional reason")
    slot = slot_arg(cfg, args[0])
    found = mode_refusal(cfg, slot, "hold") or \
        agent_slot_check(cfg, who, slot, cwd, "hold")
    if show(found, "runq hold: refused"):
        return 1
    p = slotlib.path(cfg, slot, "hold")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"by": who, "when": now(),
                   "why": " ".join(args[1:])}, f)
    os.replace(p + ".tmp", p)
    print(f"slot {slot} held: slotd starts no new job until runq go {slot}")
    return 0


def cmd_go(cfg, args, who, cwd):
    if len(args) != 1:
        raise Usage("runq go takes one slot")
    slot = slot_arg(cfg, args[0])
    found = mode_refusal(cfg, slot, "release") or \
        agent_slot_check(cfg, who, slot, cwd, "release")
    if show(found, "runq go: refused"):
        return 1
    try:
        os.remove(slotlib.path(cfg, slot, "hold"))
    except FileNotFoundError:
        print(f"slot {slot} was not held")
        return 0
    print(f"slot {slot} released: slotd starts its next job")
    return 0


# --- own, mode ---------------------------------------------------------

def cmd_own(cfg, args, who, cwd):
    if len(args) not in (2, 3):
        raise Usage('runq own takes a slot, a campaign (or none) and the '
                    'user\'s words: runq own 2 core-flashes "take slot 2"')
    slot = slot_arg(cfg, args[0])
    camp, words = args[1], (args[2] if len(args) == 3 else "").strip()
    found = mode_refusal(cfg, slot, "own")
    names = sorted(c["name"] for c in slotlib.campaigns(cfg))
    if camp != "none" and camp not in names:
        found.append(Refusal("campaign-unknown", camp, f"no campaign file "
                             f"names {camp}; the campaigns are "
                             f"{', '.join(names) or 'none'}"))
    if who.get("kind") != "user" and not words:
        found.append(Refusal("owner-words", slot, "an agent records a grant "
                             "only with the user's own words, quoted",
                             f'runq own {slot} {camp} "<the user\'s words>"'))
    if show(found, "runq own: refused"):
        return 1
    was = owner_or_none(cfg, slot)
    p = slotlib.path(cfg, slot, "owner")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"campaign": None if camp == "none" else camp,
                   "words": words, "by": who, "when": now()}, f)
    os.replace(p + ".tmp", p)
    print(f"slot {slot} owner: {camp} (was {was or 'none'})")
    return 0


def runplan_lines(cfg, slot, jobs):
    """Runplan lines for queued jobs, for a switch back to runplan mode.
    A job a runplan cannot express stays as a comment."""
    out = [f"# {now()} runq mode {slot} runplan: jobs moved back from the "
           "queue"]
    for j in jobs:
        out.append(f"# runq job {j['id']}, added by {by(j)}"
                   + (f", timeout {j['timeout_s']} s" if j.get("timeout_s")
                      else "")
                   + (f", after job {j['after']}" if j.get("after") else ""))
        if j.get("validate") or j.get("rerun"):
            flag = "--validate" if j.get("validate") else "--rerun"
            out.append(f"# not converted ({flag} has no runplan form): "
                       f"{runcheck.line(j)}")
            continue
        out.append(f"cd {shlex.quote(j['cwd'])}")
        if j.get("cmd") is not None:
            out.append(shlex.join(j["cmd"]))
            continue
        argv = j.get("argv") or []
        pin = [] if any(w.startswith("-c=") for w in argv) else \
            [f"-s={slot}"]
        case = j["case"]
        out.append(shlex.join(["sdrun.py", *pin, "-y", "-d", case, *argv])
                   + f" 2>&1 | tee {shlex.quote(case + '/BOUT.log.console')}")
    return out


def set_owner_line(text, camp, slot):
    """The runplan text with its first `# owner:` line naming camp, or with
    one added below the #! line."""
    new = f"# owner: {camp or 'none'} (set by runq mode {slot} runplan, " \
          f"{now()[:10]})"
    lines = text.split("\n")
    for k, s in enumerate(lines):
        if slotlib.OWNER_LINE.match(s):
            lines[k] = new
            return "\n".join(lines)
    k = 1 if lines and lines[0].startswith("#!") else 0
    return "\n".join(lines[:k] + [new] + lines[k:])


def cmd_mode(cfg, args, who, cwd):
    if len(args) != 2 or args[1] not in slotlib.MODES:
        raise Usage("runq mode takes a slot and queue or runplan")
    slot, want = slot_arg(cfg, args[0]), args[1]
    with slotlib.queue_lock(cfg):
        found = []
        try:
            have = slotlib.mode(cfg, slot)
        except ValueError:
            have = None    # a bad mode file is what this command repairs
        if have == want:
            print(f"slot {slot} is in {want} mode already")
            return 0
        if store_jobs(cfg, "running", slot):
            found.append(Refusal("slot-busy", slot, f"slot {slot} runs a job",
                                 f"runq cancel {slot}, or wait for it"))
        st = slotlib.read_state(cfg, slot)
        if st.get("alive") and st.get("state") not in ("idle", None):
            found.append(Refusal("slot-busy", slot, f"slotd {slot} is "
                                 f"{st.get('state')}"))
        if not slotlib.lock_free(cfg, slot):
            found.append(Refusal("slot-busy", slot, f"slot {slot}'s run lock "
                                 "is held"))
        if show(found, "runq mode: refused"):
            return 1
        plan = slotlib.runplan(cfg, slot)
        if want == "runplan":
            jobs = store_jobs(cfg, "queue", slot)
            camp = owner_or_none(cfg, slot)
            try:
                with open(plan, encoding="utf-8") as f:
                    text = f.read()
                perm = os.stat(plan).st_mode & 0o7777
            except FileNotFoundError:
                text, perm = "#!/usr/bin/env bash\nset -u\n", 0o755
            text = set_owner_line(text, camp, slot)
            if jobs:
                text = text.rstrip("\n") + "\n" + \
                    "\n".join(runplan_lines(cfg, slot, jobs)) + "\n"
            with open(plan + ".tmp", "w", encoding="utf-8") as f:
                f.write(text)
            os.chmod(plan + ".tmp", perm)
            os.replace(plan + ".tmp", plan)
            for j in jobs:
                os.remove(slotlib.job_path(cfg, "queue", slot, j["id"]))
        p = slotlib.path(cfg, slot, "mode")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            f.write(want + "\n")
        os.replace(p + ".tmp", p)
    if want == "runplan":
        print(f"slot {slot} now runs runplans; {len(jobs)} queued job"
              f"{'s' * (len(jobs) != 1)} moved to {plan}, owner "
              f"{camp or 'none'}")
    else:
        left = runcheck.runplan_cases(cfg, slot)
        print(f"slot {slot} now runs its queue; owner none until runq own "
              f"{slot}")
        if left:
            print(f"- warning: {plan} still names {len(left)} case"
                  f"{'s' * (len(left) != 1)}, which will not run in queue "
                  "mode; queue them with runq add")
    return 0


# --- restart -----------------------------------------------------------

RESTART_WAIT_S = 60   # the screen's spack setup takes several seconds


def screen_of(slot):
    """(argv, environ, cwd) of the screen session h<slot> as it was started,
    read from /proc, else None."""
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(os.path.join(d, "cmdline"), "rb") as f:
                argv = [a.decode() for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        if not argv or os.path.basename(argv[0]).lower() != "screen":
            continue
        k = next((i for i, a in enumerate(argv[:-1]) if a.startswith("-")
                  and a.endswith("S") and argv[i + 1] == f"h{slot}"), None)
        if k is None:
            continue
        try:
            with open(os.path.join(d, "environ"), "rb") as f:
                env = dict(e.decode().split("=", 1) for e in
                           f.read().split(b"\0") if b"=" in e)
            cwd = os.readlink(os.path.join(d, "cwd"))
            # screen renames itself SCREEN, so run the file it came from.
            exe = os.readlink(os.path.join(d, "exe"))
        except OSError:
            return None
        return [exe] + argv[1:], env, cwd
    return None


def cmd_restart(cfg, args, who, cwd):
    if len(args) != 1:
        raise Usage("runq restart takes one slot")
    slot = slot_arg(cfg, args[0])
    found = []
    st = slotlib.read_state(cfg, slot)
    if store_jobs(cfg, "running", slot):
        found.append(Refusal("slot-busy", slot, f"slot {slot} runs a job",
                             f"runq cancel {slot}, or wait for it"))
    elif st.get("alive") and st.get("state") not in ("idle", "held",
                                                      "stopped"):
        found.append(Refusal("slot-busy", slot, f"slotd {slot} is "
                             f"{st.get('state')}"))
    elif not slotlib.lock_free(cfg, slot):
        found.append(Refusal("slot-busy", slot, f"slot {slot}'s run lock is "
                             "held"))
    screen = screen_of(slot)
    if screen is None:
        found.append(Refusal("screen-missing", f"h{slot}", f"no screen "
                             f"session h{slot} runs, so runq cannot tell "
                             "how to start slotd; the user starts it"))
    if show(found, "runq restart: refused"):
        return 1
    argv, env, home = screen
    # The screen may have been started from an agent's shell; slotd and its
    # runs must not inherit that shell's markers.
    env = {k: v for k, v in env.items() if not runcheck.AGENT_MARK.match(k)
           and k not in ("SDTOOLS_SLOT_ENFORCE", "HERMES_SLOT", "RUNQ_JOB")}
    old = st.get("slotd_pid") if st.get("alive") else None
    if old:
        os.kill(old, signal.SIGINT)    # slotd stops; its screen then ends
        deadline = time.time() + 15
        while time.time() < deadline and (slotlib.pid_alive(old) or
                                          screen_of(slot)):
            time.sleep(0.2)
        if slotlib.pid_alive(old) or screen_of(slot):
            print(f"runq restart: slotd {slot} (pid {old}) or screen h{slot} "
                  "did not stop within 15 s; nothing started")
            return 1
    elif screen_of(slot):
        print(f"runq restart: screen h{slot} runs without a live slotd; "
              f"the user closes it (screen -S h{slot} -X quit) first")
        return 1
    subprocess.run(argv, env=env, cwd=home, check=False)
    deadline = time.time() + RESTART_WAIT_S
    while time.time() < deadline:
        new = slotlib.read_state(cfg, slot)
        if new.get("alive") and new.get("slotd_pid") != old:
            print(f"slot {slot}: slotd restarted in screen h{slot}, pid "
                  f"{new['slotd_pid']}, {new.get('mode') or 'runplan'} mode")
            return 0
        time.sleep(0.5)
    print(f"runq restart: screen h{slot} started, but no new slotd "
          f"reported within {RESTART_WAIT_S} s; check screen h{slot}")
    return 1


COMMANDS = {"add": cmd_add, "ls": cmd_ls, "rm": cmd_rm, "cancel": cmd_cancel,
            "hold": cmd_hold, "go": cmd_go, "own": cmd_own, "mode": cmd_mode,
            "restart": cmd_restart}


def main(argv=None, who=None, cwd=None):
    """Run one runq command; return its exit status. Tests pass `who` (a
    caller dict) and `cwd`; the command line never can."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__, end="")
        return 0 if argv else 2
    if argv[0] not in COMMANDS:
        print(f"runq: unknown command {argv[0]}; commands are "
              f"{', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    cfg = slotlib.config()
    if not cfg["present"]:
        print("runq: this machine has no slot config "
              "(~/.config/sdtools/slots.json), so it has no queue",
              file=sys.stderr)
        return 2
    who = who or runcheck.caller()
    try:
        return COMMANDS[argv[0]](cfg, argv[1:], who, cwd or os.getcwd())
    except (Usage, sdrun.UsageError) as e:
        print(f"runq {argv[0]}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
