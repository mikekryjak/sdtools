"""Shared slot configuration, state files and locks for slotd.py,
runlaunch.py, sdrun.py, runwait.py and runstat.py.

A workstation that runs Hermes-3 on fixed core slots describes them in a JSON
config file (default ~/.config/sdtools/slots.json, or $SDTOOLS_SLOT_CONFIG):

    {"enforce": true,
     "cases": "/home/mike/work/cases",
     "cores": {"1": "2-11", "2": "12-21", "3": "22-31"},
     "state_dir": "~/.cache/sdtools-slots",
     "campaigns": "/home/mike/work/notebooks/**/campaign.toml"}

"campaigns" is optional: a glob of TOML files, each naming a campaign
(`name`) and its source clone (`clone`). A slot's owner may then also cd
into its own clone, e.g. to run unit tests from the repo root.

Without the file nothing is enforced, so the tools behave as before on other
machines. A file that exists but does not parse, or names no cores, stops
every tool with a message: a typo must never turn enforcement off. Setting
SDTOOLS_SLOT_ENFORCE=0 lifts enforcement for one command.

Each slot's mode sits in its own file, <state_dir>/slot<N>.mode: "runplan"
(the default, when the file is absent) or "queue". In queue mode the slot's
owner sits in <state_dir>/slot<N>.owner, written by runq own; in runplan mode
it is the runplan's "# owner:" line.
"""
import fcntl
import glob
import json
import os
import re
import sys
import time

DEFAULT_CORES = {"1": "2-11", "2": "12-21", "3": "22-31"}
HEARTBEAT_S = 5      # slotd rewrites its state file at least this often
STALE_S = 30         # a state file older than this means slotd is not running
MODES = ("runplan", "queue")
OWNER_LINE = re.compile(r"(?m)^#\s*owner:\s*([A-Za-z0-9._-]*)")


def config():
    path = os.path.expanduser(os.environ.get(
        "SDTOOLS_SLOT_CONFIG", "~/.config/sdtools/slots.json"))
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        cfg = {"enforce": False, "present": False}
    except (OSError, ValueError) as e:
        sys.exit(f"slot config {path} is unreadable: {e}")
    else:
        if not isinstance(cfg, dict) or not isinstance(cfg.get("cores"), dict):
            sys.exit(f"slot config {path} names no cores")
        cfg["present"] = True
    cfg.setdefault("enforce", False)
    cfg.setdefault("cases", "/home/mike/work/cases")
    cfg.setdefault("cores", DEFAULT_CORES)
    cfg["state_dir"] = os.path.expanduser(
        cfg.get("state_dir", "~/.cache/sdtools-slots"))
    if os.environ.get("SDTOOLS_SLOT_ENFORCE") == "0":
        cfg["enforce"] = False
    return cfg


def campaigns(cfg):
    """Each campaign file's name, cases directory and clone (realpaths, or
    None when the file names none), from the config's "campaigns" glob."""
    import tomllib
    pattern = cfg.get("campaigns")
    out = []
    for p in sorted(glob.glob(os.path.expanduser(pattern), recursive=True)
                    if pattern else []):
        try:
            with open(p, "rb") as f:
                c = tomllib.load(f)
        except (OSError, ValueError):
            continue
        if not c.get("name"):
            continue
        out.append({k: os.path.realpath(os.path.expanduser(c[k]))
                    if c.get(k) else None for k in ("cases", "clone")})
        out[-1]["name"] = c["name"]
    return out


def owner_clone(cfg, owner):
    """Realpath of the clone the owner's campaign file names, or None."""
    if owner in ("", "none", None):
        return None
    return next((c["clone"] for c in campaigns(cfg)
                 if c["name"] == owner and c["clone"]), None)


def mode(cfg, slot):
    """'runplan' or 'queue', from <state_dir>/slot<N>.mode. Raises ValueError
    for any other text, so a bad file stops its slot instead of guessing."""
    p = path(cfg, slot, "mode")
    try:
        with open(p, encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        return "runplan"
    if text not in MODES:
        raise ValueError(f"{p} says {text!r}; it must say queue or runplan")
    return text


def owner(cfg, slot):
    """The campaign that owns the slot, or None. Raises ValueError when the
    mode file or the owner file is bad."""
    if mode(cfg, slot) == "queue":
        p = path(cfg, slot, "owner")
        try:
            with open(p, encoding="utf-8") as f:
                name = json.load(f).get("campaign")
        except FileNotFoundError:
            return None
        except (OSError, ValueError, AttributeError) as e:
            raise ValueError(f"{p} is unreadable: {e}")
    else:
        try:
            with open(runplan(cfg, slot), encoding="utf-8") as f:
                m = OWNER_LINE.search(f.read(4000))
        except FileNotFoundError:
            return None
        name = m.group(1) if m else None
    return None if name in ("", "none", None) else name


def cpu_set(spec):
    """Set of cores in a taskset-style list, e.g. '2-4,7' -> {2, 3, 4, 7}."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def runplan(cfg, slot):
    return os.path.join(cfg["cases"], f"runplan{slot}.sh")


def path(cfg, slot, kind):
    """State file of one slot: kind is 'state', 'lock', 'go', 'daemon',
    'mode', 'owner', 'hold' or 'cancel'."""
    os.makedirs(cfg["state_dir"], exist_ok=True)
    return os.path.join(cfg["state_dir"], f"slot{slot}.{kind}")


def read_state(cfg, slot):
    """The slot's state dict, with 'alive' added: True while slotd's
    heartbeat is fresh and its process exists."""
    try:
        with open(path(cfg, slot, "state"), encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        return {"state": "absent", "alive": False}
    fresh = time.time() - st.get("heartbeat", 0) < STALE_S
    st["alive"] = fresh and pid_alive(st.get("slotd_pid"))
    return st


def write_state(cfg, slot, st):
    st["heartbeat"] = time.time()
    tmp = path(cfg, slot, "state") + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, path(cfg, slot, "state"))


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def lock_free(cfg, slot):
    """True if no process holds the slot's run lock."""
    with open(path(cfg, slot, "lock"), "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


# The job store: one JSON file per job, under <state_dir>/<where>/<slot>/,
# where `where` is queue, running or done and `slot` is a number or "any".
# runq.py writes queue/; slotd.py moves jobs on. Every write goes to a
# temporary file first and is renamed into place, so a reader never sees
# half a job.
WHERES = ("queue", "running", "done")


def job_path(cfg, where, slot, job_id):
    d = os.path.join(cfg["state_dir"], where, str(slot))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{job_id}.json")


def write_job(cfg, where, slot, job):
    """Write `job` to <where>/<slot>/<id>.json; return its path."""
    p = job_path(cfg, where, slot, job["id"])
    tmp = os.path.join(os.path.dirname(p), f".{job['id']}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=1)
    os.replace(tmp, p)
    return p


def move_job(cfg, job_id, src, dst):
    """Move a job file from src to dst, each a (where, slot) pair, with one
    rename. Return the new path, or None when the file has gone: another
    process moved it first, so only one slotd can claim a job."""
    old = job_path(cfg, *src, job_id)
    new = job_path(cfg, *dst, job_id)
    try:
        os.rename(old, new)
    except FileNotFoundError:
        return None
    return new


class queue_lock:
    """Hold <state_dir>/queue.lock: runq holds it from its checks to its
    write, so two calls cannot both pass the duplicate check."""

    def __init__(self, cfg):
        os.makedirs(cfg["state_dir"], exist_ok=True)
        self.path = os.path.join(cfg["state_dir"], "queue.lock")

    def __enter__(self):
        self.f = open(self.path, "a")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
