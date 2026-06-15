#!/usr/bin/env python3

import argparse
import glob
import os
import shutil
import sys

from boutdata.collect import collect

"""
reset_test.py - Reset a case to its baseline restart files.

A test/perf case folder is expected to contain a `base` subdirectory holding a
known set of BOUT++ restart files (BOUT.restart.*). This tool copies those
restart files up into the case directory, overwriting any existing ones, so the
case is reset to its baseline starting point ready to be re-run.

Usage:
    reset_test.py <case folder>
"""


def read_sim_time(path):
    """Return (t, seconds) from the restart files in path, or None.

    t is the simulation time in normalised units; the time normalisation is
    1/Omega_ci, so physical seconds is t / Omega_ci.
    """
    try:
        t = float(collect("tt", path=path, prefix="BOUT.restart", info=False))
        omega_ci = float(collect("Omega_ci", path=path, prefix="BOUT.restart", info=False))
        return t, t / omega_ci
    except Exception:
        return None


def format_time(time):
    if time is None:
        return "unknown (no readable restart files)"
    _, seconds = time
    return f"{seconds * 1e3:.6g} ms"


def reset_test(casepath):
    casepath = os.path.abspath(casepath)
    basepath = os.path.join(casepath, "base")

    if not os.path.isdir(basepath):
        raise FileNotFoundError(
            f"No 'base' directory found in case '{casepath}'.\n"
            f"Expected baseline restart files at: {basepath}"
        )

    restarts = sorted(glob.glob(os.path.join(basepath, "BOUT.restart.*")))
    if not restarts:
        raise FileNotFoundError(
            f"No BOUT++ restart files (BOUT.restart.*) found in '{basepath}'."
        )

    # Simulation time currently in the case (before we overwrite it).
    prev_time = read_sim_time(casepath)

    for src in restarts:
        shutil.copy2(src, os.path.join(casepath, os.path.basename(src)))

    new_time = read_sim_time(casepath)

    print(f"Reset {casepath}: copied {len(restarts)} restart file(s) from {basepath}")
    print(f"  Previous simulation time: {format_time(prev_time)}")
    print(f"  New simulation time:      {format_time(new_time)}")


# ------------------------------------------------------------
# PARSER
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reset a case to the baseline restart files in its 'base' directory."
    )
    parser.add_argument("casepath", help="Path to the case folder")
    args = parser.parse_args()

    reset_test(args.casepath)
