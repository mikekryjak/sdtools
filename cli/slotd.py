#!/usr/bin/env python3
"""Resident launcher for one core slot. Run it in that slot's screen.

Usage: slotd.py <slot> [--no-screen-check]

Waits for a trigger written by runlaunch.py, then runs runplan<slot>.sh from
the cases directory, pinned to the slot's cores with taskset and holding the
slot's run lock, with HERMES_SLOT=<slot> in its environment. sdrun.py refuses
to start without that variable once the slot config enforces it, so a run can
only start through this loop. The runplan's output appears in the screen.

State, heartbeat and last exit code go to <state_dir>/slot<N>.state, which
runlaunch.py, runwait.py and runstat.py read. Ctrl-C during a run stops the
run and returns to waiting; Ctrl-C while waiting stops slotd.
"""
import fcntl
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import slotlib  # noqa: E402

POLL_S = 1
TRIGGER_MAX_AGE_S = 15   # runlaunch.py waits this long; an older trigger is stale


def log(slot, msg):
    print(f"[slotd {slot} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_once(cfg, slot, trigger, st):
    plan = slotlib.runplan(cfg, slot)
    lock = open(slotlib.path(cfg, slot, "lock"), "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log(slot, "run lock is held by another process; launch refused")
        st.update(last_error="run lock held", last_launch=trigger.get("id"))
        return
    cores = cfg["cores"][str(slot)]
    env = dict(os.environ, HERMES_SLOT=str(slot))
    env.pop("SDTOOLS_SLOT_ENFORCE", None)   # a manual override never reaches runs
    log(slot, f"launch {trigger.get('id')}: {plan} on cores {cores}")
    # The runplan inherits the lock, so the slot stays locked while it or
    # anything it started in the background lives, even if slotd dies.
    child = subprocess.Popen(["taskset", "-c", cores, "bash", plan],
                             cwd=cfg["cases"], env=env,
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
    daemon = open(slotlib.path(cfg, slot, "daemon"), "a")
    try:
        fcntl.flock(daemon, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"slotd {slot} is already running")
    go = slotlib.path(cfg, slot, "go")
    if os.path.exists(go):
        os.remove(go)   # a trigger left from before this slotd started is stale
    st = dict(state="idle", slotd_pid=os.getpid(), started=time.time(),
              runplan=slotlib.runplan(cfg, slot))
    slotlib.write_state(cfg, slot, st)
    log(slot, f"waiting for runlaunch.py {slot}")
    try:
        while True:
            orphan = not slotlib.lock_free(cfg, slot)
            st["state"] = "orphan" if orphan else "idle"
            if os.path.exists(go):
                try:
                    with open(go, encoding="utf-8") as f:
                        trigger = json.load(f)
                except (OSError, ValueError):
                    trigger = {"id": "unreadable"}
                os.remove(go)
                if time.time() - trigger.get("time", 0) > TRIGGER_MAX_AGE_S:
                    log(slot, f"ignored stale trigger {trigger.get('id')}")
                elif orphan:
                    log(slot, "an earlier runplan still holds the run lock; "
                        "launch refused")
                    st.update(last_error="earlier runplan still running",
                              last_launch=trigger.get("id"))
                else:
                    run_once(cfg, slot, trigger, st)
                    log(slot, f"waiting for runlaunch.py {slot}")
            slotlib.write_state(cfg, slot, st)
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        pass
    st.update(state="stopped", pid=None)
    slotlib.write_state(cfg, slot, st)
    log(slot, "stopped")


if __name__ == "__main__":
    main()
