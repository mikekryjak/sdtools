#!/usr/bin/env python3

import argparse
import csv
import getpass
import os
import re
import subprocess
from datetime import datetime

"""
record_test.py - Record one perftest run as a row in a CSV.

For a case directory it reads run metadata from the case's BOUT.log file
(originator, run start/finish time, wall-clock duration, Hermes-3 commit with
branch and date from the local git repo, and BOUT++ version and commit), and
describes the solver settings relative to a named base recipe.

The base recipe is given with --recipe PATH (a path to the recipe file). Every
option under [solver] and [petsc] in the case's BOUT.inp is compared with that
recipe; each option that differs, is missing, or is extra is recorded in the
'diffs' column as "section:key: <recipe value> -> <case value>". The recipe is
identified in the CSV by its file name without extension.

Solver settings are read from BOUT.inp (the input the run used) rather than
BOUT.settings, which dumps every resolved default and so produces large,
uninformative diffs against the partial recipes. Run metadata is read from
BOUT.log.*.

Usage:
    record_test.py <case folder> --recipe recipes/SNES-MUMPS-1.txt [--csv path] [--note "..."]
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "run_records.csv")

# Settings sections compared against the recipe.
RECIPE_SECTIONS = ("solver", "petsc")

CSV_COLUMNS = [
    "case", "originator", "run_started", "run_time_str", "run_time_s",
    "recipe", "diffs", "note",
    "hermes_branch", "hermes_commit", "hermes_commit_date",
    "bout_version", "bout_commit", "recorded_at",
]

_LOG_TIME_FMT = "%a %b %d %H:%M:%S %Y"


# ------------------------------------------------------------
# Log parsing (from test_results.py)
# ------------------------------------------------------------
def find_log(casepath):
    logs = [f for f in os.listdir(casepath) if re.fullmatch(r"BOUT\.log\.\d+", f)]
    if not logs:
        return None
    logs.sort(key=lambda f: int(f.rsplit(".", 1)[1]))
    return os.path.join(casepath, logs[0])


def _search(pattern, text, group=1):
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(group).strip() if m else None


def parse_log(logpath):
    with open(logpath, "r", errors="replace") as fh:
        text = fh.read()

    out = {
        "bout_version": _search(r"^BOUT\+\+ version (.+)$", text),
        "bout_commit": _search(r"^Revision:\s*([0-9a-fA-F]+)", text),
        "hermes_commit": _search(r"^Git Version of Hermes:\s*([0-9a-fA-F]+)", text),
        "run_started": _search(r"^Run started at\s*:\s*(.+)$", text),
        "run_finished": _search(r"^Run finished at\s*:\s*(.+)$", text),
        "run_time_str": _search(r"^Run time\s*:\s*(.+)$", text),
        "binary": _search(r"Command line options for this run :\s*(\S+)", text),
    }

    out["wall_seconds"] = None
    if out["run_started"] and out["run_finished"]:
        try:
            t0 = datetime.strptime(out["run_started"], _LOG_TIME_FMT)
            t1 = datetime.strptime(out["run_finished"], _LOG_TIME_FMT)
            out["wall_seconds"] = (t1 - t0).total_seconds()
        except ValueError:
            pass
    return out


def find_repo_root(start_path):
    if not start_path:
        return None
    p = os.path.abspath(start_path)
    if os.path.isfile(p):
        p = os.path.dirname(p)
    while True:
        if os.path.exists(os.path.join(p, ".git")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def _git(repo, *args):
    try:
        res = subprocess.run(["git", "-C", repo, *args],
                             capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_ok(repo, *args):
    try:
        subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def git_commit_date(repo, commit):
    """Author/commit date of a commit, or None if it can't be resolved.

    The branch a run used is deliberately not inferred from the commit: a
    commit can live on many branches and is the tip of none, so the branch
    is taken from --branch (or the current checkout) instead."""
    if not repo or not commit:
        return None
    return _git(repo, "show", "-s", "--format=%ci", commit)


def current_branch(repo):
    """The repo's currently checked-out branch, or None if detached/unknown."""
    if not repo:
        return None
    head = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")  # branch name or "HEAD"
    return head if head and head != "HEAD" else None


# ------------------------------------------------------------
# Recipe / settings parsing and diffing
# ------------------------------------------------------------
def parse_ini(text, sections):
    """Minimal INI parser. Returns {section: {key: value}} for the given
    sections. Inline '#' comments are stripped; a key with no value (a bare
    PETSc flag) is treated as 'true'."""
    wanted = set(sections)
    out = {s: {} for s in sections}
    section = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section not in wanted:
            continue
        line = line.split("#", 1)[0].strip()  # drop inline comment
        if not line:
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            out[section][key.strip()] = val.strip()
        else:
            out[section][line] = "true"  # bare flag
    return out


def _norm_key(key):
    # BOUT++ option names are case- and underscore-insensitive, e.g.
    # pid_controller <-> pidController.
    return key.strip().lower().replace("_", "")


def _norm_val(val):
    val = val.strip()
    low = val.lower()
    if low in ("true", "false"):
        return low
    try:
        return repr(float(val))  # 1e-6 == 1e-06 == 0.000001, 0 == 0.0
    except ValueError:
        return low


def load_recipe(recipe_path):
    if not os.path.isfile(recipe_path):
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")
    with open(recipe_path) as fh:
        return parse_ini(fh.read(), RECIPE_SECTIONS)


def load_case_input(casepath):
    """Return {section: {key: value}} of the [solver]/[petsc] options the run
    used, read from BOUT.inp."""
    inp = os.path.join(casepath, "BOUT.inp")
    if not os.path.isfile(inp):
        raise FileNotFoundError(f"No BOUT.inp in '{casepath}'.")
    with open(inp, "r", errors="replace") as fh:
        return parse_ini(fh.read(), RECIPE_SECTIONS)


def diff_recipe(recipe, case):
    """Compare every [solver]/[petsc] option in the case against the recipe.
    Returns a list of 'section:key: <recipe> -> <case>' strings covering options
    that changed, are unset in the case, or are extra (not in the recipe)."""
    diffs = []
    for section in RECIPE_SECTIONS:
        rkeys = recipe.get(section, {})
        ckeys = case.get(section, {})
        cnorm = {_norm_key(k): (k, v) for k, v in ckeys.items()}
        rnorm = {_norm_key(k): k for k in rkeys}

        # Recipe keys that were changed or unset in the case.
        for nk, rk in rnorm.items():
            rv = rkeys[rk]
            if nk in cnorm:
                cv = cnorm[nk][1]
                if _norm_val(rv) != _norm_val(cv):
                    diffs.append(f"{section}:{rk}: {rv} -> {cv}")
            else:
                diffs.append(f"{section}:{rk}: {rv} -> (unset)")

        # Case keys that the recipe does not mention.
        for nk, (ck, cv) in cnorm.items():
            if nk not in rnorm:
                diffs.append(f"{section}:{ck}: (none) -> {cv}")
    return diffs


# ------------------------------------------------------------
# Record assembly
# ------------------------------------------------------------
def record_test(casepath, recipe_path, note="", branch=None):
    casepath = os.path.abspath(casepath)
    logpath = find_log(casepath)
    if logpath is None:
        raise FileNotFoundError(f"No BOUT.log.* file found in {casepath}")

    info = parse_log(logpath)
    hermes_repo = find_repo_root(info["binary"])
    commit_date = git_commit_date(hermes_repo, info["hermes_commit"])
    # The branch isn't recorded in BOUT.log, so it can't be recovered from the
    # commit. Use the caller-supplied --branch, else the repo's current checkout.
    hermes_branch = branch or current_branch(hermes_repo)

    recipe = load_recipe(recipe_path)
    # Identify the recipe in the CSV by its file name, without extension.
    recipe_name = os.path.splitext(os.path.basename(recipe_path))[0]
    case_opts = load_case_input(casepath)
    diffs = diff_recipe(recipe, case_opts)

    row = {
        "case": os.path.basename(casepath.rstrip(os.sep)),
        "originator": _originator(hermes_repo),
        "run_started": info["run_started"] or "",
        "run_time_str": info["run_time_str"] or "",
        "run_time_s": "" if info["wall_seconds"] is None else int(round(info["wall_seconds"])),
        "recipe": recipe_name,
        "diffs": "; ".join(diffs),
        "note": note,
        "hermes_branch": hermes_branch or "",
        "hermes_commit": info["hermes_commit"] or "",
        "hermes_commit_date": commit_date or "",
        "bout_version": info["bout_version"] or "",
        "bout_commit": info["bout_commit"] or "",
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return row


def _originator(repo=None):
    """Email of the active git account (repo-local config wins over global),
    falling back to the OS login name if git has no email configured."""
    email = _git(repo, "config", "user.email") if repo else None
    if not email:
        email = _git(os.getcwd(), "config", "user.email")  # global / cwd repo
    return email or getpass.getuser()


def append_csv(row, csv_path):
    # Write a header if the file is missing or empty (an existing empty file,
    # e.g. one created by opening it in an editor, still needs a header).
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def print_row(row):
    width = max(len(c) for c in CSV_COLUMNS)
    for col in CSV_COLUMNS:
        print(f"  {col:<{width}} : {row[col]}")


# ------------------------------------------------------------
# PARSER
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a perftest run (metadata + solver recipe diff) as a CSV row."
    )
    parser.add_argument("casepath", help="Path to the case folder")
    parser.add_argument("--recipe", required=True,
                        help="Path to the base solver recipe file")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"CSV file to append to (default: ./{DEFAULT_CSV} in the current directory)")
    parser.add_argument("--note", default="", help="Optional free-text note for this run")
    parser.add_argument("--branch",
                        help="Hermes-3 branch the run was on (not recoverable from "
                             "the log). Defaults to the hermes repo's current checkout.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the row but do not write to the CSV")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip the branch confirmation prompt")
    args = parser.parse_args()

    row = record_test(args.casepath, args.recipe, note=args.note, branch=args.branch)
    print_row(row)
    if args.dry_run:
        print("\n(dry run: nothing written)")
    else:
        # Confirm the branch before writing: it can't be checked against the log,
        # so a wrong --branch (or stale checkout) would silently be recorded.
        if not args.yes:
            shown = row["hermes_branch"] or "(unknown)"
            reply = input(f"\nRecord this run as hermes branch '{shown}'? [y/N] ")
            if reply.strip().lower() not in ("y", "yes"):
                print("Aborted: nothing written.")
                raise SystemExit(1)
        append_csv(row, args.csv)
        print(f"\nAppended to {os.path.abspath(args.csv)}")
