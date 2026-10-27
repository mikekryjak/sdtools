#!/usr/bin/env python3
"""Wait until the runs queued in one or more runplans have ended. Read-only.

Usage: runwait.py <slot> [<slot> ...] [--stall-min N] [--max-s S]

Watches the cases on the live (uncommented) sdrun.py lines of
runplan<slot>.sh. Returns when none is running for IDLE_S seconds, when one
stalls (no write, or no new simulated time in BOUT.log.0, for --stall-min
minutes, default 60), or at the deadline (--max-s seconds, default 3600).
Prints why it returned (done, stalled or deadline), then one
"<state>\t<case>" line per case. A second wait on a watched runplan exits at
once, so one wait covers each slot.

Run it as a background task after confirming a launch; after a deadline,
check the runs, then start it again.

Environment, for tests: RUNWAIT_RUNPLAN (default
/home/mike/work/cases/runplan{}.sh), RUNWAIT_LOCKDIR (default
~/.cache/runwait), RUNWAIT_IDLE_S (default 180).
"""
import argparse
import fcntl
import glob
import hashlib
import os
import re
import sys
import time

RUNPLAN = os.environ.get("RUNWAIT_RUNPLAN",
                         "/home/mike/work/cases/runplan{}.sh")
LOCKDIR = os.environ.get("RUNWAIT_LOCKDIR",
                         os.path.expanduser("~/.cache/runwait"))
IDLE_S = int(os.environ.get("RUNWAIT_IDLE_S", 180))  # quiet this long: done
DASH_D = re.compile(r"\s-d[= ]\s*(\S+)")
SIM_ROW = re.compile(r"(?m)^\s*(\d\.\d+e[+-]\d+)\s+\d+\s")  # BOUT.log.0 output


def cases_in(text, base):
    """Cases on the live sdrun.py lines of a runplan's text, resolved from
    its last cd, else from base."""
    cds = re.findall(r"(?m)^\s*cd\s+(\S+)", text)
    base = cds[-1] if cds else base
    out = []
    for line in text.splitlines():
        d = DASH_D.search(line)
        if d and "sdrun.py" in line and not line.lstrip().startswith("#"):
            out.append(os.path.realpath(os.path.join(base, d.group(1))))
    return out


def queued(slot):
    """Cases queued in runplan<slot>.sh; [] if it cannot be read."""
    try:
        text = open(RUNPLAN.format(slot), encoding="utf-8").read()
    except OSError:
        return []
    return cases_in(text, os.path.dirname(RUNPLAN.format(slot)))


def lock(runplans):
    """Hold one lock per runplan path for this process's life; exit if
    another wait holds one."""
    os.makedirs(LOCKDIR, exist_ok=True)
    held = []
    for path in map(os.path.realpath, runplans):
        key = hashlib.sha256(path.encode()).hexdigest()[:16]
        f = open(os.path.join(LOCKDIR, key + ".lock"), "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit(f"already watched: another wait holds {path}")
        held.append(f)
    return held


def running(case):
    """True while some process holds the case's console open: `tee` keeps it
    open for exactly the life of the run's pipeline."""
    console = os.path.realpath(os.path.join(case, "BOUT.log.console"))
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) == console:
                return True
        except OSError:
            continue
    return False


def last_write(case):
    files = glob.glob(os.path.join(case, "BOUT.dmp.*")) + [
        os.path.join(case, n) for n in ("BOUT.log.0", "BOUT.log.console")]
    times = [os.path.getmtime(f) for f in files if os.path.isfile(f)]
    return max(times) if times else 0


def sim_time(case):
    """The simulated time of the last output row in BOUT.log.0, or None."""
    log = os.path.join(case, "BOUT.log.0")
    if not os.path.isfile(log):
        return None
    with open(log, "rb") as f:
        f.seek(max(0, os.path.getsize(log) - 262144))
        rows = SIM_ROW.findall(f.read().decode(errors="replace"))
    return rows[-1] if rows else None


def status(case, stall_min=60):
    log = os.path.join(case, "BOUT.log.0")
    console = os.path.join(case, "BOUT.log.console")
    if os.path.isfile(log) and re.search(r"(?m)^Run time",
                                         open(log, errors="replace").read()):
        return "finished"
    if running(case):
        if time.time() - last_write(case) > 60 * stall_min:
            return "stalled"
        return "running"
    if os.path.isfile(console) or os.path.isfile(log):
        return "stopped"
    return "queued"


def wait_cases(cases_fn, stall_min, deadline, idle_s=IDLE_S):
    """Poll cases_fn() each minute until none runs for idle_s, one stalls, or
    the deadline passes. Return (why, {case: state})."""
    idle_since = None
    progress = {}   # case -> (last simulated time, when it last changed)
    while True:
        cases = cases_fn()
        states = {c: status(c, stall_min) for c in cases}
        now = time.time()
        for c in [c for c, s in states.items() if s == "running"]:
            t, since = progress.get(c, (None, now))
            new = sim_time(c) or t   # a row pushed out of the tail is no news
            if new != t:
                since = now
            progress[c] = (new, since)
            if now - since > 60 * stall_min:
                states[c] = "stalled"
        if any(s == "stalled" for s in states.values()):
            return "stalled", states
        if not any(s == "running" for s in states.values()):
            idle_since = idle_since or now
            if now - idle_since > idle_s or not cases:
                return "done", states
        else:
            idle_since = None
        if now >= deadline:
            return "deadline", states
        time.sleep(min(60, idle_s))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("slots", nargs="+", type=int)
    p.add_argument("--stall-min", type=int, default=60)
    p.add_argument("--max-s", type=int, default=3600)
    a = p.parse_args()
    held = lock([RUNPLAN.format(n) for n in a.slots])  # noqa: F841
    why, states = wait_cases(
        lambda: [c for n in a.slots for c in queued(n)],
        a.stall_min, time.time() + a.max_s)
    print(why)
    for c, s in sorted(states.items()):
        print(f"{s}\t{c}")


if __name__ == "__main__":
    main()
