#!/usr/bin/env python3
"""Start a slot's runplan through its slotd. The one launch route for agents.

Usage: runlaunch.py <slot> [--check]

Refuses, and starts nothing, unless every check passes:
  - exactly the configured runplans exist (runplan1.sh .. runplanN.sh in the
    cases directory, and no runplan?.sh below it);
  - the runplan's "# owner:" line names an owner other than none, and every
    cd target and every sdrun.py case path holds the owner as a path segment
    (a cd may also enter the owner's clone, from the config's "campaigns");
  - it queues at least one command line;
  - every sdrun.py line runs on this slot (-s=<slot>, or -c inside its cores)
    and tees to <case>/BOUT.log.console;
  - no sdrun.py line without -restart or -append targets a case whose
    BOUT.log.0 already has a "Run time" (re-running it would destroy it);
  - slotd for the slot is alive and idle, and the slot's run lock is free.
On a slot in queue mode it refuses, and prints a `runq add` line for each
live runplan line instead.
Then it writes the trigger and waits for slotd to confirm. --check runs the
checks and stops there. Exit status: 0 launched (or checks passed), 1 refused,
2 slotd did not confirm in time.
"""
import fcntl
import glob
import json
import os
import re
import shlex
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import slotlib  # noqa: E402

CONFIRM_S = 15
SDRUN_D = re.compile(r"\s-d[=\s]\s*(\S+)")
# Words that would move a run off its slot or around slotd's checks.
ESCAPES = re.compile(r"HERMES_SLOT|SDTOOLS_SLOT_ENFORCE|\btaskset\b|"
                     r"\bmpirun\b|\bmpiexec\b|hermes3v")


def live_lines(text):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("set ") or \
                s.startswith("#!"):
            continue
        out.append(s)
    return out


def problems(cfg, slot):
    """Every reason not to launch slot `slot`, as a list of strings."""
    errs = []
    cases = cfg["cases"]
    plan = slotlib.runplan(cfg, slot)
    expected = {os.path.realpath(slotlib.runplan(cfg, n)) for n in cfg["cores"]}
    found = set()
    for depth in ("", "*/", "*/*/", "*/*/*/"):
        found |= {os.path.realpath(p)
                  for p in glob.glob(os.path.join(cases, depth + "runplan?.sh"))}
    for extra in sorted(found - expected):
        errs.append(f"an extra runplan exists: {extra}; only "
                    f"{sorted(expected)} may exist")
    if not os.path.isfile(plan):
        return errs + [f"{plan} does not exist"]
    text = open(plan, encoding="utf-8").read()

    m = re.search(r"(?m)^#\s*owner:\s*([A-Za-z0-9._-]*)", text)
    owner = m.group(1) if m else ""
    if owner in ("", "none"):
        errs.append(f"slot {slot} has no owner (owner line: "
                    f"'{owner or 'missing'}'); only the user assigns a slot")

    lines = [part for s in live_lines(text)
             for part in re.split(r"\s*(?:&&|;)\s*", s) if part]
    root = cases
    clone = slotlib.owner_clone(cfg, owner)
    cmds = []
    for s in lines:
        if ESCAPES.search(s):
            errs.append(f"line sets the slot, pinning or MPI itself, or runs "
                        f"hermes3v; queue plain sdrun.py lines: {s}")
        cd = re.match(r"(?:cd|pushd)\s+(\S+)", s)
        if cd:
            root = os.path.realpath(os.path.join(root, cd.group(1)))
            if owner not in ("", "none") and \
                    f"/{owner}/" not in root + "/" and root != clone:
                errs.append(f"cd target {root} is not inside '{owner}', "
                            "the slot's owner")
            continue
        cmds.append((s, root))
    if not cmds:
        errs.append(f"{plan} queues no command line")

    mine = slotlib.cpu_set(cfg["cores"][str(slot)])
    for s, base in cmds:
        if "sdrun.py" not in s:
            continue
        d = SDRUN_D.search(s)
        case = os.path.realpath(os.path.join(base, d.group(1))) if d else None
        sflag = re.search(r"\s-s[=\s]\s*(\S+)", s)
        cflag = re.search(r"\s-c[=\s]\s*(\S+)", s)
        if cflag:
            if not slotlib.cpu_set(cflag.group(1)) <= mine:
                errs.append(f"-c={cflag.group(1)} leaves slot {slot}'s "
                            f"cores {cfg['cores'][str(slot)]}: {s}")
        elif not sflag or sflag.group(1) != str(slot):
            errs.append(f"sdrun.py line is not on slot {slot}: {s}")
        if not re.search(r"tee\s+(?:-a\s+)?\S*BOUT\.log\.console", s):
            errs.append(f"sdrun.py line does not tee to "
                        f"<case>/BOUT.log.console: {s}")
        if case is None:
            errs.append(f"sdrun.py line names no case with -d: {s}")
            continue
        if owner not in ("", "none") and f"/{owner}/" not in case + "/":
            errs.append(f"case {case} is not inside '{owner}'")
        if not re.search(r"\s-?(restart|append)\b", s):
            log0 = os.path.join(case, "BOUT.log.0")
            if os.path.isfile(log0) and re.search(
                    r"(?m)^Run time", open(log0, errors="replace").read()):
                errs.append(f"{case} has already finished; a scratch re-run "
                            "would destroy it. Record it and delete its line.")

    st = slotlib.read_state(cfg, slot)
    if not st["alive"]:
        errs.append(f"slotd {slot} is not running in screen h{slot} "
                    f"(state: {st.get('state')})")
    elif st.get("state") != "idle":
        errs.append(f"slot {slot} is busy: state {st.get('state')}, "
                    f"pid {st.get('pid')}")
    if not slotlib.lock_free(cfg, slot):
        errs.append(f"slot {slot}'s run lock is held")
    return errs


def queue_lines(cfg, slot):
    """A `runq add` line for each live line of the slot's runplan, or a
    note for a line runq cannot express."""
    try:
        text = open(slotlib.runplan(cfg, slot), encoding="utf-8").read()
    except OSError:
        return []
    root, out = cfg["cases"], []
    for s in [p for line in live_lines(text)
              for p in re.split(r"\s*(?:&&|;)\s*", line) if p]:
        cd = re.match(r"(?:cd|pushd)\s+(\S+)$", s)
        if cd:
            root = os.path.realpath(os.path.join(root, cd.group(1)))
            continue
        head = re.split(r"\s*(?:2>&1|\||>)", s)[0]
        try:
            words = shlex.split(head)
        except ValueError:
            words = []
        timeout = []
        if words[:1] == ["timeout"] and len(words) > 2:
            timeout, words = ["--timeout", words[1]], words[2:]
        if words and words[0].endswith("sdrun.py"):
            case, rest, i = None, [], 1
            while i < len(words):
                w = words[i]
                if w in ("-d", "-s") and i + 1 < len(words):
                    case = words[i + 1] if w == "-d" else case
                    i += 2
                    continue
                if w.startswith("-d="):
                    case = w[3:]
                elif not (w.startswith("-s=") or w in ("-y", "--yes")):
                    rest.append(w)
                i += 1
            if case:
                case = os.path.realpath(os.path.join(root, case))
                cwd = [] if root == os.path.dirname(case) else ["--cwd", root]
                out.append("runq add " + shlex.join(
                    [slot, case, *rest, *cwd, *timeout]))
                continue
        elif words and not re.search(r"[|<>&$`]", s):
            out.append("runq add " + shlex.join(
                [slot, "--cwd", root, *timeout, "--", *words]))
            continue
        out.append(f"# cannot convert, queue it by hand: {s}")
    return out


def main():
    args = sys.argv[1:]
    check_only = "--check" in args
    args = [a for a in args if a != "--check"]
    if len(args) != 1:
        sys.exit(__doc__)
    slot = args[0]
    cfg = slotlib.config()
    if slot not in cfg["cores"]:
        sys.exit(f"unknown slot {slot}; slots are {sorted(cfg['cores'])}")
    try:
        mode = slotlib.mode(cfg, slot)
    except ValueError as e:
        sys.exit(f"runlaunch {slot}: refused: {e}")
    if mode == "queue":
        print(f"runlaunch {slot}: refused, nothing started: slot {slot} runs "
              f"its queue, not runplan{slot}.sh. Queue the runplan's lines "
              "with runq (an agent's job also needs --timeout):")
        for line in queue_lines(cfg, slot) or ["(the runplan has no live line)"]:
            print(f"  {line}")
        sys.exit(1)
    # One launch per slot at a time, from the checks to slotd's answer.
    held = open(slotlib.path(cfg, slot, "launch"), "a")
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"runlaunch {slot}: another launch of this slot is in progress")
    errs = problems(cfg, slot)
    if errs:
        print(f"runlaunch {slot}: refused, nothing started")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    if check_only:
        print(f"runlaunch {slot}: checks pass")
        return
    launch = uuid.uuid4().hex[:12]
    go = slotlib.path(cfg, slot, "go")
    with open(go + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"id": launch, "time": time.time(), "by": os.getppid()}, f)
    os.replace(go + ".tmp", go)
    deadline = time.time() + CONFIRM_S
    while time.time() < deadline:
        st = slotlib.read_state(cfg, slot)
        if st.get("launch") == launch and st.get("state") == "running":
            print(f"runlaunch {slot}: started {slotlib.runplan(cfg, slot)}, "
                  f"runplan pid {st.get('pid')}, launch {launch}")
            return
        if st.get("launch") == launch:
            # A short runplan can finish between two reads of the state.
            print(f"runlaunch {slot}: started and already ended, exit "
                  f"{st.get('last_exit')}, launch {launch}")
            return
        if st.get("last_launch") == launch and st.get("last_error"):
            print(f"runlaunch {slot}: slotd refused: {st['last_error']}")
            sys.exit(1)
        time.sleep(0.5)
    try:
        with open(go, encoding="utf-8") as f:
            if json.load(f).get("id") == launch:
                os.remove(go)   # never leave a trigger to fire later
    except (OSError, ValueError):
        pass
    print(f"runlaunch {slot}: slotd did not confirm within {CONFIRM_S} s; "
          f"its trigger was withdrawn if still pending. Check screen h{slot}")
    sys.exit(2)


if __name__ == "__main__":
    main()
