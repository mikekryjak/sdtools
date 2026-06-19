#!/usr/bin/env python3

import argparse
import glob
import os
import sys

from boutdata.collect import collect

"""
get_sim_times.py - Print the simulation time reached by each matching case.

Given one or more name patterns (shell-style wildcards), find the case
directories that match, read the simulation time from each case's
BOUT.restart.* files, and print it in milliseconds.

The time normalisation is 1/Omega_ci, so physical seconds is tt / Omega_ci.

Usage:
    get_sim_times.py 'test*'        # all cases with names starting 'test'
    get_sim_times.py test2 test4    # specific cases (or shell-expanded glob)
"""


def read_sim_time(path):
    """Return the latest simulation time in seconds for a case, or None.

    Reads the simulation time (tt) and Omega_ci from the case's BOUT.restart.*
    files; the time normalisation is 1/Omega_ci, so physical seconds is
    tt / Omega_ci.
    """
    try:
        tt = float(collect("tt", path=path, prefix="BOUT.restart", info=False))
        omega_ci = float(
            collect("Omega_ci", path=path, prefix="BOUT.restart", info=False)
        )
        return tt / omega_ci
    except Exception:
        return None


def find_cases(patterns):
    """Expand the patterns to a sorted, de-duplicated list of directories."""
    matches = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            full = os.path.abspath(path)
            if os.path.isdir(path) and full not in seen:
                seen.add(full)
                matches.append(path)
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Print the simulation time of each case matching a pattern."
    )
    parser.add_argument(
        "patterns",
        nargs="+",
        help="Case name pattern(s), shell-style wildcards (quote to avoid shell expansion)",
    )
    args = parser.parse_args()

    cases = find_cases(args.patterns)
    if not cases:
        print(f"No case directories match: {' '.join(args.patterns)}", file=sys.stderr)
        sys.exit(1)

    width = max(len(os.path.basename(c.rstrip(os.sep))) for c in cases)
    for case in cases:
        name = os.path.basename(case.rstrip(os.sep))
        seconds = read_sim_time(case)
        if seconds is None:
            timestr = "no data (no readable BOUT.restart.* files)"
        else:
            timestr = f"{seconds * 1e3:.6g} ms"
        print(f"  {name:<{width}} : {timestr}")


if __name__ == "__main__":
    main()
