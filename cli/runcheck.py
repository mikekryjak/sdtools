"""Checks on one queued run, shared by runq.py, when it adds a job, and by
slotd.py, when it starts one.

A job is a dict, saved as one JSON file in a slot's queue. make_job() builds
it from the words after `runq add <slot>`:

    <case> [sdrun.py options] [runq options]       a Hermes-3 case
    [runq options] -- <command> [args]             any other job

runq options: --cwd DIR, --timeout T (e.g. 90s, 30m, 10h), --after JOB,
--study S<nn>, --validate, --rerun. A case job runs from the case's parent
unless --cwd says otherwise, as a runplan's `cd` did, so relative grid paths
resolve. Relative case and -p paths resolve from --cwd, else from where runq
was called.

problems() returns every reason not to add or start a job, as Refusal
records. Each names its rule, a stable id that a stuck-agent limit can count,
and, where one exists, the command that would pass. A Refusal with warn=True
is shown but stops nothing.
"""
import dataclasses
import datetime as dt
import glob
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import runwait  # noqa: E402
import sdrun  # noqa: E402
import slotlib  # noqa: E402

# Environment variables that mark an agent's shell. Claude Code sets
# CLAUDECODE and CLAUDE_CODE_*; Codex sets no fixed marker, so a caller
# without a terminal is an agent too.
AGENT_MARK = re.compile(r"CLAUDE|CODEX|AI_AGENT")
# Words in a case job's options that would move it off its slot.
ESCAPES = re.compile(r"HERMES_SLOT|SDTOOLS_SLOT_ENFORCE|\btaskset\b|"
                     r"\bmpirun\b|\bmpiexec\b|hermes3v")
# Words in an agent's `--` command that would start a run without the case
# checks. `hermes-3/` (a directory) is not a binary.
AGENT_DENY = re.compile(r"(?:^|[\s/'\"=;&|(])(?:sdrun\.py|sdrun|hermes-3|"
                        r"mpirun|mpiexec|hermes3v|taskset)(?=$|[\s'\";&|)])|"
                        r"HERMES_SLOT|SDTOOLS_SLOT_ENFORCE")
FILE_KEY = re.compile(r"""(?mi)^\s*(?:grid|file)\s*=\s*("[^"]*"|'[^']*'|[^\s#]+)""")
NC_VALUE = re.compile(r"""(?m)^\s*[\w:]+\s*=\s*("[^"\n]*\.nc"|'[^'\n]*\.nc')""")
UNUSED_OFF = re.compile(r"(?mi)^\s*error_on_unused_options\s*=\s*false\b")
JOB_OPTS = ("--cwd", "--after", "--timeout", "--study")
SDRUN_VALUED = ("-s", "-c", "-np", "-b", "-p", "--path")
VALIDATE_TIMEOUT_S = 900
HERMES_ROOT = "/home/mike/work/hermes-3"   # sdrun.py's default for -b


@dataclasses.dataclass
class Refusal:
    rule: str          # stable id, e.g. "spent-case"
    target: str        # the case, slot, file or command it is about
    message: str       # what is wrong, in one sentence
    fix: str = ""      # the command that would pass, if one exists
    warn: bool = False  # True: shown, but the job goes ahead

    def __str__(self):
        head = ("warning: " if self.warn else "") + self.message
        return head + (f"\n  fix: {self.fix}" if self.fix else "")


def caller(env=None, tty=None):
    """Who is calling: {"kind": "user"} only from a terminal with no agent
    marker in the environment; {"kind": "job", ...} from inside a job slotd
    started; else an agent, with its session id and tmux session name."""
    env = os.environ if env is None else env
    if env.get("RUNQ_JOB"):
        return {"kind": "job", "job": env["RUNQ_JOB"]}
    tty = sys.stdin.isatty() if tty is None else tty
    if tty and not any(AGENT_MARK.match(k) for k in env):
        return {"kind": "user"}
    tmux = None
    if env.get("TMUX"):
        try:
            tmux = subprocess.run(["tmux", "display-message", "-p", "#S"],
                                  capture_output=True, text=True,
                                  timeout=2).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    return {"kind": "agent", "session": env.get("CLAUDE_CODE_SESSION_ID"),
            "tmux": tmux}


_last_stamp = [""]


def new_id():
    """A job id that sorts by the time it was added, to the microsecond, so
    the lines of one batch keep their order."""
    t = dt.datetime.now()
    stamp = t.strftime("%Y%m%d-%H%M%S-%f")
    while stamp <= _last_stamp[0]:
        t += dt.timedelta(microseconds=1)
        stamp = t.strftime("%Y%m%d-%H%M%S-%f")
    _last_stamp[0] = stamp
    return stamp + "-" + secrets.token_hex(2)


def parse_timeout(text):
    m = re.fullmatch(r"(\d+)([smhd]?)", text.strip())
    if not m or int(m.group(1)) < 1:
        raise ValueError(f"--timeout {text}: give a whole number of s, m, h "
                         "or d, e.g. 10h")
    return int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600,
                              "d": 86400}[m.group(2)]


def expand(value):
    return os.path.expanduser(os.path.expandvars(value))


def inside(path, root):
    return bool(root) and (path == root or path.startswith(root.rstrip("/") + "/"))


def linked_libs(exe):
    """The shared libraries `exe` loads from its own build directory, such as
    libhermes-3-lib.so and libbout++.so, which a rebuild may change while
    the executable stays the same."""
    try:
        out = subprocess.run(["ldd", exe], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    build = os.path.realpath(os.path.dirname(exe))
    libs = {os.path.realpath(p) for p in re.findall(r"=>\s*(/\S+)", out)}
    return sorted(p for p in libs if inside(p, build))


def binary_hash(exe):
    """sha256 over the executable and its build-local libraries."""
    h = hashlib.sha256()
    for p in [exe] + linked_libs(exe):
        h.update(p.encode() + b"\0")
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def campaign_of(cfg, path):
    """The one campaign whose cases directory or clone holds `path`, else
    None."""
    names = {c["name"] for c in slotlib.campaigns(cfg)
             if inside(path, c["cases"]) or inside(path, c["clone"])}
    return names.pop() if len(names) == 1 else None


def make_job(cfg, slot, args, cwd=None, who=None):
    """A job dict from the words after `runq add <slot>`. Raises ValueError
    on a command line that names no job."""
    cwd = cwd or os.getcwd()
    job = {"id": new_id(), "slot": str(slot), "by": who or caller(),
           "added": dt.datetime.now().isoformat(timespec="seconds"),
           "args": list(args), "campaign": None, "study": None,
           "mandate": None, "cwd": None, "case": None, "argv": [],
           "slot_flag": None, "cmd": None, "exe": None, "exe_hash": None,
           "binary_named": False, "validate": False, "rerun": False,
           "after": None, "timeout_s": None}
    rest, given_cwd, i = [], None, 0
    while i < len(args):
        a = args[i]
        name, eq, val = a.partition("=")
        if a == "--":
            job["cmd"] = list(args[i + 1:])
            break
        if name in JOB_OPTS:
            if not eq:
                if i + 1 >= len(args):
                    raise ValueError(f"{a} needs a value")
                val = args[i + 1]
                i += 1
            if name == "--cwd":
                given_cwd = val
            elif name == "--after":
                job["after"] = val
            elif name == "--timeout":
                job["timeout_s"] = parse_timeout(val)
            else:
                job["study"] = val
        elif a == "--validate":
            job["validate"] = True
        elif a == "--rerun":
            job["rerun"] = True
        else:
            rest.append(a)
        i += 1
    base = os.path.join(cwd, expand(given_cwd)) if given_cwd else cwd

    if job["cmd"] is not None:
        if rest:
            raise ValueError(f"before -- give only runq options, not {rest}")
        if not job["cmd"]:
            raise ValueError("give a command after --")
        if job["validate"] or job["rerun"]:
            raise ValueError("--validate and --rerun apply to a case, not a "
                             "command after --")
        job["cwd"] = os.path.realpath(base)
        job["campaign"] = campaign_of(cfg, job["cwd"])
        return job

    case_arg, words, i = None, [], 0
    while i < len(rest):
        a = rest[i]
        if a == "-d" and i + 1 < len(rest):
            case_arg = rest[i + 1]
            i += 2
            continue
        if a.startswith("-d="):
            case_arg = a[3:]
        elif case_arg is None and not a.startswith("-") and \
                not (words and words[-1] in SDRUN_VALUED):
            case_arg = a
        else:
            words.append(a)
        i += 1
    if case_arg is None:
        raise ValueError("name a case, or give a command after --")
    opts = sdrun.parse_args(words)      # raises sdrun.UsageError
    if opts["help"]:
        raise ValueError("-h is not a job; run sdrun.py -h for its options")
    case = os.path.realpath(os.path.join(base, expand(case_arg)))
    job["case"] = case
    job["cwd"] = os.path.realpath(base) if given_cwd else os.path.dirname(case)
    job["binary_named"] = any(w in ("-p", "--path", "-b") or
                              w.startswith(("-p=", "--path=", "-b="))
                              for w in words)
    if opts["exe_path"]:
        job["exe"] = os.path.realpath(os.path.join(base,
                                                   expand(opts["exe_path"])))
    elif job["binary_named"]:
        root = os.environ.get("hermes", HERMES_ROOT)
        job["exe"] = os.path.realpath(os.path.join(root, expand(opts["build"]),
                                                   "hermes-3"))
    argv = []
    if opts["core_spec"]:
        argv.append(f"-c={opts['core_spec']}")
    if opts["nranks"]:
        argv.append(f"-np={opts['nranks']}")
    if job["exe"]:
        argv.append(f"-p={job['exe']}")
    if opts["pass_restart"]:
        argv.append("-restart")
    if opts["pass_append"]:
        argv.append("-append")
    job["argv"] = argv + opts["extra_args"]
    job["slot_flag"] = opts["slot"] or None
    if job["exe"] and os.path.isfile(job["exe"]):
        job["exe_hash"] = binary_hash(job["exe"])
    if job["validate"]:
        job["timeout_s"] = VALIDATE_TIMEOUT_S
    job["campaign"] = campaign_of(cfg, case)
    return job


def line(job, add=(), drop=()):
    """The runq command for this job, with words added or dropped."""
    words = [a for a in job["args"] if a not in drop]
    if "--" in words:
        k = words.index("--")
        words = words[:k] + list(add) + words[k:]
    else:
        words += list(add)
    return f"runq add {job['slot']} {shlex.join(words)}"


def spent(case):
    """True once a case has any output: a run that ended at a timeout, a
    cancel or a stall has no "Run time" line but has overwritable output."""
    return os.path.isfile(os.path.join(case, "BOUT.log.0")) or \
        bool(glob.glob(os.path.join(case, "BOUT.dmp.*")))


def input_files(text):
    """The file paths BOUT.inp names: grid and file keys, and quoted *.nc
    values. Values built from ${...} or $VAR are skipped."""
    vals = FILE_KEY.findall(text) + NC_VALUE.findall(text)
    out = []
    for v in vals:
        v = v.strip("\"'")
        if v and "$" not in v and v not in out:
            out.append(v)
    return out


def jobs(cfg, where=("queue", "running")):
    """(where, job) for every job file in those parts of the store."""
    for w in where:
        for p in sorted(glob.glob(os.path.join(cfg["state_dir"], w, "*",
                                               "*.json"))):
            try:
                with open(p, encoding="utf-8") as f:
                    yield w, json.load(f)
            except (OSError, ValueError):
                continue


def find_job(cfg, job_id):
    """(where, job) of the job with this id, else (None, None)."""
    return next(((w, j) for w, j in jobs(cfg, ("queue", "running", "done"))
                 if j.get("id") == job_id), (None, None))


def runplan_cases(cfg, n):
    try:
        with open(slotlib.runplan(cfg, n), encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []
    return runwait.cases_in(text, cfg["cases"])


def problems(cfg, job, when, slot=None):
    """Every reason not to add (when="add") or start (when="start", on
    `slot`) this job, as a list of Refusal."""
    out = []

    def no(rule, target, message, fix="", warn=False):
        out.append(Refusal(rule, str(target), message, fix, warn))

    slot = str(slot or job["slot"])
    who = job.get("by") or {}
    agent = who.get("kind") == "agent"
    if slot != "any" and slot not in cfg["cores"]:
        no("slot-unknown", slot, f"slot {slot} does not exist; the slots are "
           f"{', '.join(sorted(cfg['cores']))} and any")
        return out
    slots = sorted(cfg["cores"]) if slot == "any" else [slot]
    modes = {}
    for n in slots:
        try:
            modes[n] = slotlib.mode(cfg, n)
        except ValueError as e:
            no("slot-mode", n, str(e))
    queue_slots = [n for n in slots if modes.get(n) == "queue"]
    if slot != "any" and modes.get(slot) == "runplan":
        no("slot-mode", slot, f"slot {slot} runs runplans until the user "
           "switches it to queue mode", f"queue it in runplan{slot}.sh")
    elif slot == "any" and len(modes) == len(slots) and not queue_slots:
        no("slot-mode", "any", "no slot is in queue mode yet")
    if when == "add" and who.get("kind") == "job":
        no("nested-job", who.get("job"), "a job queues no jobs",
           "chain the jobs instead: runq add <slot> ... --after <job>")

    deferred = when == "add" and bool(job.get("after"))
    case, argv = job.get("case"), job.get("argv") or []
    if case:
        inp = os.path.join(case, "BOUT.inp")
        if not deferred:
            if not os.path.isfile(inp):
                no("case-missing", case, f"{case} holds no BOUT.inp")
            else:
                text = open(inp, encoding="utf-8", errors="replace").read()
                if UNUSED_OFF.search(text):
                    no("unused-options", case, f"{inp} sets "
                       "error_on_unused_options = false, so a misspelt "
                       "option passes unseen", warn=True)
                missing = [f for f in input_files(text) if not os.path.exists(
                    os.path.join(job["cwd"], f))]
                if missing:
                    d, fix = case, ""
                    while d != "/":
                        if all(os.path.exists(os.path.join(d, f))
                               for f in missing):
                            fix = line(job, add=["--cwd", d])
                            break
                        d = os.path.dirname(d)
                    no("file-unresolved", case, f"{', '.join(missing)}, "
                       f"named in {inp}, does not exist from the job's "
                       f"directory {job['cwd']}", fix)
        exe = job.get("exe")
        if not job.get("binary_named"):
            no("binary-unnamed", case, "name the binary with -p=<path> or "
               "-b=<build>; sdrun.py's default build-mc-master is never "
               "used unasked", line(job, add=["-p=<path to hermes-3>"]))
        elif not deferred and not (exe and os.path.isfile(exe) and
                                   os.access(exe, os.X_OK)):
            no("binary-missing", exe, f"no executable exists at {exe}")
        elif when == "start" and job.get("exe_hash") and \
                os.path.isfile(exe) and binary_hash(exe) != job["exe_hash"]:
            no("binary-changed", exe, f"{exe} or one of its build's "
               "libraries changed since the job was added",
               "add the job again: " + line(job))
        sflag = job.get("slot_flag")
        if sflag and sflag != slot:
            no("slot-flag", case, f"-s={sflag} names another slot than "
               f"{slot}; runq sets the slot itself",
               line(job, drop=[f"-s={sflag}"]))
        cflag = next((w[3:] for w in argv if w.startswith("-c=")), None)
        if cflag and slot == "any":
            no("slot-flag", case, f"-c={cflag} names cores, so it needs a "
               "slot number, not any")
        elif cflag and not slotlib.cpu_set(cflag) <= \
                slotlib.cpu_set(cfg["cores"][slot]):
            no("slot-flag", case, f"-c={cflag} leaves slot {slot}'s cores "
               f"{cfg['cores'][slot]}")
        esc = [w for w in argv if ESCAPES.search(w)]
        if esc:
            no("escape", case, f"{esc[0]} would move the run off its slot or "
               "around slotd")
        restart, append = "-restart" in argv, "-append" in argv
        if job.get("validate"):
            pass   # a validate pass runs in a scratch copy of the case
        elif job.get("rerun") and (restart or append):
            no("rerun-restart", case, "--rerun starts the case afresh, so it "
               "takes no -restart or -append",
               line(job, drop=["-restart", "-append", "restart", "append"]))
        elif not deferred and spent(case) and not append and \
                not job.get("rerun"):
            if restart:
                no("restart-only", case, f"{case} has output, and -restart "
                   "alone starts new dump files over it; continue it with "
                   "-append", line(job, add=["-append"]))
            else:
                no("spent-case", case, f"{case} has run already (it holds "
                   "BOUT.log.0 or BOUT.dmp.*); a fresh run would overwrite "
                   "it. Continue it, or rerun it with its old output moved "
                   "to overwritten-<time>/",
                   f"{line(job, add=['-restart', '-append'])}   or   "
                   f"{line(job, add=['--rerun'])}")
        if when == "add" and not job.get("validate"):
            for w, other in jobs(cfg):
                if other.get("id") not in (job["id"], job.get("after")) and \
                        other.get("case") == case and \
                        not other.get("validate"):
                    no("duplicate", case, f"{case} is already {w} as job "
                       f"{other['id']} on slot {other.get('slot')}",
                       "runq ls")
            for n in sorted(cfg["cores"]):
                if case in runplan_cases(cfg, n):
                    no("duplicate", case, f"{case} is already queued in "
                       f"runplan{n}.sh")
    elif job.get("cmd") is not None:
        if not deferred and not os.path.isdir(job["cwd"]):
            no("cwd-missing", job["cwd"], f"{job['cwd']} does not exist")
        text = " ".join(job["cmd"])
        if agent and AGENT_DENY.search(text):
            no("agent-command", text, "an agent queues a Hermes-3 run as a "
               "case job, so the case checks run",
               f"runq add {slot} <case> -p=<path to hermes-3> "
               "--timeout <limit>")

    if job.get("after"):
        w, prior = find_job(cfg, job["after"])
        if when == "add" and w is None:
            no("after-unknown", job["after"], f"no job {job['after']} exists "
               "to wait for", "runq ls")
        elif when == "start" and w != "done":
            no("after-pending", job["after"], f"job {job['after']} has not "
               "ended yet")
        elif when == "start" and prior.get("outcome") != "ok":
            no("after-failed", job["after"], f"job {job['after']} ended "
               f"{prior.get('outcome')}, so this job is skipped")

    if agent:
        if job.get("timeout_s") is None:
            no("agent-timeout", job.get("case") or job.get("cwd"),
               "an agent's job gives its own timeout, as a backstop for a "
               "hung run", line(job, add=["--timeout", "10h"]))
        camp = job.get("campaign")
        where = job.get("case") or job.get("cwd")
        if camp is None:
            no("agent-campaign", where, f"{where} lies inside no single "
               "campaign's cases directory or clone")
        else:
            owners = {}
            for n in (queue_slots if slot == "any" else [slot]):
                try:
                    owners[n] = slotlib.owner(cfg, n)
                except ValueError as e:
                    no("slot-mode", n, str(e))
            grant = ("ask the user for a slot, then record the grant: runq "
                     f"own <slot> {camp} \"<the user's words>\"")
            if slot != "any" and slot in owners and owners[slot] != camp:
                no("agent-owner", slot, f"slot {slot} belongs to "
                   f"{owners[slot] or 'no campaign'}, not {camp}", grant)
            elif slot == "any" and camp not in owners.values():
                no("agent-owner", "any", f"{camp} owns no slot in queue "
                   "mode", grant)

    m = job.get("mandate")
    if when == "start" and m and (
            not os.path.isfile(m.get("state", "")) or
            dt.datetime.fromisoformat(m["expires"]) <= dt.datetime.now()):
        no("mandate-ended", m.get("state"), "the mandate this job was added "
           "under has ended, so the job does not start")
    return out
