"""Command-line entry point for caseplan.

Subcommands:

- ``caseplan gen cases.csv``   -> casegen (create/update cases)
- ``caseplan index STUDY``     -> refresh status columns of cases.csv
- ``caseplan init STUDY``      -> bootstrap cases.csv for an existing study
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from . import index as index_mod
from . import init as init_mod
from .generate import GenerationError, GenOptions, casegen


def _add_gen(sub):
    p = sub.add_parser("gen", help="create/update cases from cases.csv (casegen)")
    p.add_argument("cases", help="path to cases.csv")
    p.add_argument("--dry-run", action="store_true", help="print actions and diffs, change nothing")
    p.add_argument("--case", help="generate only this case")
    p.add_argument("--update-inputs-only", action="store_true",
                   help="update inputs for an existing generated case without outputs")
    p.add_argument("--runs-dir", default="runs", help="runs directory (default: runs)")
    p.add_argument("--legacy-flat", action="store_true",
                   help="create cases as direct study children")
    p.add_argument("--force-existing", action="store_true",
                   help="allow updating an existing generated case without outputs")
    p.add_argument("--allow-output-dir", action="store_true",
                   help="allow touching a generated case that contains outputs (rare)")
    p.add_argument("--overwrite-handmade", action="store_true",
                   help="allow touching a directory without case.json (rare)")
    return p


def _add_index(sub):
    p = sub.add_parser("index", help="refresh status columns of cases.csv")
    p.add_argument("study", help="study directory")
    p.add_argument("--runs-dir", default="runs", help="runs directory (default: runs)")
    p.add_argument("--legacy-flat", action="store_true",
                   help="treat direct study children as cases")
    return p


def _add_init(sub):
    p = sub.add_parser("init", help="bootstrap cases.csv for an existing study")
    p.add_argument("study", help="study directory")
    p.add_argument("--base", help="infer diffs against this base case/path")
    p.add_argument("--runs-dir", default="runs", help="runs directory (default: runs)")
    p.add_argument("--legacy-flat", action="store_true",
                   help="treat direct study children as cases")
    p.add_argument("--smart-diff", dest="smart_diff", action="store_true", default=True,
                   help="infer likely parents by minimal option differences (default)")
    p.add_argument("--no-smart-diff", dest="smart_diff", action="store_false",
                   help="only list observed cases and evidence")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing cases.csv")
    return p


def _run_gen(args) -> int:
    opts = GenOptions(
        dry_run=args.dry_run,
        case=args.case,
        update_inputs_only=args.update_inputs_only,
        runs_dir=args.runs_dir,
        legacy_flat=args.legacy_flat,
        force_existing=args.force_existing,
        allow_output_dir=args.allow_output_dir,
        overwrite_handmade=args.overwrite_handmade,
    )
    try:
        results = casegen(args.cases, opts)
    except GenerationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for r in results:
        head = f"== {r.name}"
        if r.skipped:
            head += " (skipped)"
        print(head)
        for m in r.messages:
            print(m)
    return 0


def _run_index(args) -> int:
    rows = index_mod.refresh(args.study, runs_dir=args.runs_dir, legacy_flat=args.legacy_flat)
    print(index_mod.format_table(rows))
    print(f"\nwrote {Path(args.study) / 'cases.csv'}")
    return 0


def _run_init(args) -> int:
    try:
        csv_path = init_mod.init_study(
            args.study,
            base=args.base,
            runs_dir=args.runs_dir,
            legacy_flat=args.legacy_flat,
            smart_diff=args.smart_diff,
            force=args.force,
        )
    except init_mod.InitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {csv_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="caseplan", description=__doc__)
    parser.add_argument("--version", action="version", version=f"caseplan {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_gen(sub)
    _add_index(sub)
    _add_init(sub)

    args = parser.parse_args(argv)
    if args.command == "gen":
        return _run_gen(args)
    if args.command == "index":
        return _run_index(args)
    if args.command == "init":
        return _run_init(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
