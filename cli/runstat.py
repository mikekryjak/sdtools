#!/usr/bin/env python3
"""Live status of Hermes-3 runs and runplan queues. Read-only.

Reconciles three sources, no bookkeeping required:
  1. ps            -> which cases are RUNNING, on which exe
  2. CPU affinity  -> which slot (s1: 2-11, s2: 12-21, s3: 22-31)
  3. runplan*.sh   -> the queue, with each line classified live:
                      RUNNING / DONE / PARTIAL n/nout / NOT STARTED
     (commented lines are skipped, so stale comments can't lie; an
      uncommented line whose case is already DONE is flagged.)

Usage: runstat.py [CASES_DIR]              live status (default: cwd)
       runstat.py --history [N] [CASES_DIR]   last N runs, oldest first
                                              (default N=15; N=0 -> all)

Progress comes from each case's BOUT.log.0 (output lines seen vs nout in
BOUT.settings) and the exe's CHECK level from the log header. History is
reconstructed from the logs ("Run started at", command line, log mtime), so
it needs no ledger and covers past runs — but a re-run overwrites its dir's
log, so history shows the LAST run of each case dir.
"""

import datetime
import os
import re
import glob
import subprocess
import sys

SLOTS = {"2-11": "s1", "12-21": "s2", "22-31": "s3"}


def running_procs():
    """{case_dir: dict(exe=, pid=, slot=)} for live hermes-3 ranks (rank 0)."""
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    procs = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\S*hermes-3)\s+-d\s+(\S+)", line)
        if not m:
            continue
        pid, exe, case = m.group(1), m.group(2), os.path.abspath(m.group(3))
        if case in procs:
            continue  # keep first rank only
        slot = "?"
        try:
            with open(f"/proc/{pid}/status") as f:
                cpus = re.search(r"Cpus_allowed_list:\s*(\S+)", f.read()).group(1)
            slot = SLOTS.get(cpus, cpus)
        except Exception:
            pass
        procs[case] = dict(exe=exe, pid=pid, slot=slot)
    return procs


def case_progress(case_dir):
    """(n_outputs_done, nout, last_sim_time, check_level) from log + settings."""
    n, last_t, nout, check = 0, None, None, "?"
    log = os.path.join(case_dir, "BOUT.log.0")
    if os.path.exists(log):
        with open(log, errors="replace") as f:
            for line in f:
                m = re.match(r"([\d.]+e[+-]\d+)\s+\d+\s+[\d.]", line)
                if m:
                    n += 1
                    last_t = m.group(1)
                elif "error checking enabled, level" in line:
                    check = line.rsplit("level", 1)[1].strip()
                elif "error checking disabled" in line:
                    check = "0"
    st = os.path.join(case_dir, "BOUT.settings")
    if os.path.exists(st):
        with open(st, errors="replace") as f:
            for line in f:
                m = re.match(r"nout\s*=\s*(\d+)", line)
                if m:
                    nout = int(m.group(1))
                    break
    return max(0, n - 1), nout, last_t, check  # first output line is t=0


def classify(case_dir, procs):
    if case_dir in procs:
        return "RUNNING"
    done, nout, _, _ = case_progress(case_dir)
    if nout is None:
        return "NOT STARTED" if done == 0 else f"PARTIAL {done}/?"
    return "DONE" if done >= nout else f"PARTIAL {done}/{nout}"


def run_record(case_dir, procs):
    """Parse one case's last run from its BOUT.log.0; None if never ran."""
    log = os.path.join(case_dir, "BOUT.log.0")
    if not os.path.exists(log):
        return None
    start, exe, restart = None, "?", ""
    with open(log, errors="replace") as f:
        for line in f:
            if start is None and "Run started at" in line:
                stamp = " ".join(line.split(":", 1)[1].split())
                try:
                    start = datetime.datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y")
                except ValueError:
                    pass
            elif "Command line options for this run" in line:
                m = re.search(r":\s*(\S+)", line)
                exe = m.group(1) if m else "?"
                if re.search(r"\brestart\b", line):
                    restart = " restart"
            if start is not None and exe != "?":
                break
    done, nout, t, check = case_progress(case_dir)
    running = case_dir in procs
    end = datetime.datetime.now() if running else \
        datetime.datetime.fromtimestamp(os.path.getmtime(log))
    hours = (end - start).total_seconds() / 3600 if start else float("nan")
    status = "RUNNING" if running else ("done" if nout and done >= nout
                                        else f"partial {done}/{nout or '?'}")
    return dict(case=os.path.basename(case_dir), start=start, hours=hours,
                status=status, frac=f"{done}/{nout or '?'}", t=t, check=check,
                exe=exe.split("/work/")[-1] + restart)


def show_history(root, n_last, procs):
    recs = []
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        r = run_record(os.path.abspath(d.rstrip("/")), procs)
        if r and r["start"]:
            recs.append(r)
    recs.sort(key=lambda r: r["start"])
    if n_last:
        recs = recs[-n_last:]
    print(f"== last {len(recs)} runs (oldest first; re-runs hide their predecessors) ==")
    for r in recs:
        print(f"  {r['start']:%m-%d %H:%M}  {r['hours']:5.1f}h  {r['status']:>14s}  "
              f"CHECK={r['check']:>2s}  {r['case']:55s} {r['exe']}")


def main():
    args = sys.argv[1:]
    n_last = 15
    if args and args[0] == "--history":
        args.pop(0)
        if args and args[0].isdigit():
            n_last = int(args.pop(0))
        root = os.path.abspath(args[0] if args else ".")
        show_history(root, n_last, running_procs())
        return
    root = os.path.abspath(args[0] if args else ".")
    procs = running_procs()

    print("== RUNNING ==")
    if not procs:
        print("  (nothing)")
    for case, p in sorted(procs.items(), key=lambda kv: kv[1]["slot"]):
        done, nout, t, check = case_progress(case)
        frac = f"{done}/{nout}" if nout else f"{done}/?"
        print(f"  {p['slot']}  {os.path.basename(case):55s} {frac:>10s}  t={t}  "
              f"CHECK={check}  exe={p['exe'].split('/work/')[-1]}")

    for plan in sorted(glob.glob(os.path.join(root, "runplan*.sh"))):
        print(f"\n== {os.path.basename(plan)} ==")
        any_line = False
        for line in open(plan):
            m = re.match(r"\s*(#?)\s*echo y \| sdrun\.py\s+(.*-d\s+(\S+).*)", line)
            if not m:
                continue
            commented, case = bool(m.group(1)), m.group(3)
            if commented:
                continue
            any_line = True
            cd = case if os.path.isabs(case) else os.path.join(root, case)
            status = classify(os.path.abspath(cd), procs)
            slot = (re.search(r"-s=(\d)", m.group(2)) or [None, "?"])[1]
            note = "   <- done but line not commented out" if status == "DONE" else ""
            print(f"  s{slot}  {status:18s} {case}{note}")
        if not any_line:
            print("  (no pending lines)")


if __name__ == "__main__":
    main()
