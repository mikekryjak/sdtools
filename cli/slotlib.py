"""Shared slot configuration, state files and locks for slotd.py,
runlaunch.py, sdrun.py, runwait.py and runstat.py.

A workstation that runs Hermes-3 on fixed core slots describes them in a JSON
config file (default ~/.config/sdtools/slots.json, or $SDTOOLS_SLOT_CONFIG):

    {"enforce": true,
     "cases": "/home/mike/work/cases",
     "cores": {"1": "2-11", "2": "12-21", "3": "22-31"},
     "state_dir": "~/.cache/sdtools-slots"}

Without the file nothing is enforced, so the tools behave as before on other
machines. Setting SDTOOLS_SLOT_ENFORCE=0 lifts enforcement for one command.
"""
import fcntl
import json
import os
import time

DEFAULT_CORES = {"1": "2-11", "2": "12-21", "3": "22-31"}
HEARTBEAT_S = 5      # slotd rewrites its state file at least this often
STALE_S = 30         # a state file older than this means slotd is not running


def config():
    path = os.path.expanduser(os.environ.get(
        "SDTOOLS_SLOT_CONFIG", "~/.config/sdtools/slots.json"))
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["present"] = True
    except (OSError, ValueError):
        cfg = {"enforce": False, "present": False}
    cfg.setdefault("enforce", False)
    cfg.setdefault("cases", "/home/mike/work/cases")
    cfg.setdefault("cores", DEFAULT_CORES)
    cfg["state_dir"] = os.path.expanduser(
        cfg.get("state_dir", "~/.cache/sdtools-slots"))
    if os.environ.get("SDTOOLS_SLOT_ENFORCE") == "0":
        cfg["enforce"] = False
    return cfg


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
    """State file of one slot: kind is 'state', 'lock', 'go' or 'daemon'."""
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
