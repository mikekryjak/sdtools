#!/usr/bin/env python3
"""Re-derive stored records from their raw artefacts and report any difference.

Checks that today's code still produces what is on disk. It cannot tell you a
stored number was right in the first place -- only that nothing has silently
changed since it was written.

    verify_store.py --store /path/to/results-repo [--cases DIR ...] [--limit N]

Exits non-zero if any record differs. Rows whose case directory has been
deleted are skipped and counted, since there is nothing left to re-derive from.
"""

import argparse
import csv
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from perftest.verify import rederive  # noqa: E402

DEFAULT_ROOTS = [
    "/home/mike/work/cases/perftests/test2dev",
    "/home/mike/work/cases/perftests/test4dev",
    "/home/mike/work/cases/perftests/test5dev",
]


def find_case(case_dir, roots):
    """The directory a row was run in, or None.

    The empty check is not defensive padding. A legacy row carries no case_dir,
    and os.path.join(root, "") is the root itself -- which exists, so without
    this the row "verifies" against a directory that is not a case at all and
    reports a match because an empty record has nothing to disagree with.
    """

    if not (case_dir or "").strip():
        return None
    for root in roots:
        path = os.path.join(root, case_dir.strip())
        if os.path.isdir(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", required=True, help="results store")
    parser.add_argument("--cases", nargs="*", default=DEFAULT_ROOTS,
                        help="directories to look for case directories in")
    parser.add_argument("--limit", type=int, help="check at most this many rows")
    parser.add_argument("--rtol", type=float, default=1e-9,
                        help="relative tolerance for numeric comparison")
    args = parser.parse_args()

    index = os.path.join(args.store, "index.tsv")
    with open(index, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    live = [(r, find_case(r.get("case_dir", ""), args.cases)) for r in rows]
    skipped = sum(1 for _, p in live if p is None)
    live = [(r, p) for r, p in live if p]
    if args.limit:
        live = live[:args.limit]

    failed = 0
    for row, path in live:
        diffs = rederive(path, args.store, row, rtol=args.rtol)
        label = row.get("test_id") or row.get("case_dir")
        if diffs:
            failed += 1
            print(f"DIFFERS  {label}")
            for d in diffs[:8]:
                print(f"    {d}")
            if len(diffs) > 8:
                print(f"    ... and {len(diffs) - 8} more")
        else:
            print(f"ok       {label}")

    print(f"\n{len(live)} checked, {len(live) - failed} match, {failed} differ,"
          f" {skipped} skipped (case directory deleted)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
