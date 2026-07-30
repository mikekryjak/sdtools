#!/usr/bin/env python3
"""Extract finished Hermes-3 performance tests into a results store.

Reads a case directory, writes the bundle and one index row, and says whether
the extraction validated -- which is the only thing that licenses deleting the
dumps. Nothing is transcribed by hand.

    extract_test.py <case-dir> [<case-dir> ...] --store /path/to/results-repo

Exits non-zero if any case failed to extract, so a queue of them can be trusted.
"""

import argparse
import pathlib
import sys

# sdtools ships no packaging; cli/ is on $PATH and its tools are run by name.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perftest import extract_case  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="+", help="case directories to extract")
    parser.add_argument(
        "--store", required=True, help="results store (holds index.tsv and runs/)"
    )
    parser.add_argument(
        "--grid",
        help="grid file, if it is not named in BOUT.inp or not beside the case",
    )
    parser.add_argument(
        "--conduction",
        help="conduction method, e.g. Harmonic. Has no dump constant although the"
        " slope limiter does, so it is the one build option needing declaring",
    )
    parser.add_argument("--epoch", help="epoch this run belongs to")
    parser.add_argument(
        "--recipes",
        help="directory of recipe files. Without it, `diffs` is not computed and"
        " a drifted recipe goes unnoticed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be recorded, write nothing",
    )
    args = parser.parse_args()

    failed = 0
    for case in args.cases:
        report = extract_case(
            case,
            args.store,
            grid_path=args.grid,
            recipes_dir=args.recipes,
            conduction_method=args.conduction,
            epoch=args.epoch,
            dry_run=args.dry_run,
        )
        print(report)
        if args.dry_run:
            for key in sorted(report.record):
                print(f"    {key:20s} {report.record[key]}")
        print()
        if not report.ok:
            failed += 1

    if failed:
        print(f"{failed} of {len(args.cases)} case(s) did not validate.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
