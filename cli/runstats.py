#!/usr/bin/env python3

import argparse
import getpass
import os
import platform
import re
import subprocess
from datetime import datetime

"""
runstats.py - Summarise how/when/where a Hermes-3 run was produced.

Reads metadata recorded in a simulation directory's BOUT.log.* file:
    - run start / finish time and wall-clock duration
    - Hermes-3 commit
    - BOUT++ version and commit

These fields are written into the log by BOUT++/Hermes-3 at runtime and need
nothing outside the simulation directory.

Some requested fields are NOT stored in the run directory:
    - the originator (user@host) the run was performed on (never logged)
    - the git *branch* and Hermes-3 commit *date* (only hashes are logged)

The originator is read from the machine running this script, assuming it is the
same user and host that ran the simulation.

The branch and commit date can be recovered with --git, which looks the logged
hashes up in the local git repositories (the Hermes-3 source dir is inferred
from the binary path recorded in the log; BOUT++ is taken from
<hermes>/external/BOUT-dev). This steps outside the run directory and is marked
as such in the output.

Usage:
    runstats.py [casepath]        # default: current directory
    runstats.py [casepath] --git  # also resolve branches + commit dates
"""

# Log datetime format, e.g. "Mon Jun 15 11:49:28 2026"
_LOG_TIME_FMT = "%a %b %d %H:%M:%S %Y"


def find_log(casepath):
    """Return the path to BOUT.log.0, or the lowest-numbered BOUT.log.* present."""
    logs = [f for f in os.listdir(casepath) if re.fullmatch(r"BOUT\.log\.\d+", f)]
    if not logs:
        return None
    logs.sort(key=lambda f: int(f.rsplit(".", 1)[1]))
    return os.path.join(casepath, logs[0])


def _search(pattern, text, group=1):
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(group).strip() if m else None


def parse_log(logpath):
    """Extract run metadata from a BOUT.log file. Directory-only information."""
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

    # Exact wall-clock seconds from the start/finish stamps (cross-check on the
    # human-readable "Run time" string, and works even if the run was killed).
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
    """Walk up from a path until a directory containing .git is found."""
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
        res = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True, text=True, check=True,
        )
        return res.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_lookup(repo, commit):
    """Best-effort branch + exact commit date for a hash in a local repo.

    NOTE: this reads the local git repo, not the run directory. The branch is
    ambiguous in general (a commit can live on many branches) and is reported
    best-effort; the commit date is exact when the commit is present locally.
    """
    if not repo or not commit:
        return {"branch": None, "date": None, "repo": repo}

    date = _git(repo, "show", "-s", "--format=%ci", commit)

    branch = None
    head = _git(repo, "rev-parse", "HEAD")
    full = _git(repo, "rev-parse", commit)
    if head and full and head == full:
        cur = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if cur and cur != "HEAD":  # "HEAD" means detached (common for submodules)
            branch = cur
    if not branch:
        contains = _git(repo, "branch", "--contains", commit, "--format=%(refname:short)")
        if contains:
            names = [b.strip() for b in contains.splitlines()
                     if b.strip() and not b.strip().startswith("(")]  # drop detached-HEAD line
            branch = ", ".join(names) if names else None

    return {"branch": branch, "date": date, "repo": repo}


def _fmt_duration(seconds):
    if seconds is None:
        return None
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def runstats(casepath, use_git=True):
    casepath = os.path.abspath(casepath)
    logpath = find_log(casepath)
    if logpath is None:
        raise FileNotFoundError(f"No BOUT.log.* file found in {casepath}")

    info = parse_log(logpath)
    info["casepath"] = casepath
    info["logfile"] = os.path.basename(logpath)
    # The originator (user@host) is never written into the run directory, so we
    # read it from the machine running this script, assuming it is the same user
    # and host that ran the simulation.
    user = getpass.getuser()
    host = platform.node()
    info["originator"] = f"{user}@{host}" if (user and host) else (user or host or None)
    info["hermes_git"] = {"branch": None, "date": None, "repo": None}
    info["bout_git"] = {"branch": None, "date": None, "repo": None}

    if use_git:
        hermes_repo = find_repo_root(info["binary"])
        info["hermes_git"] = git_lookup(hermes_repo, info["hermes_commit"])
        bout_repo = None
        if hermes_repo:
            cand = os.path.join(hermes_repo, "external", "BOUT-dev")
            bout_repo = find_repo_root(cand) if os.path.isdir(cand) else None
        info["bout_git"] = git_lookup(bout_repo, info["bout_commit"])

    return info


def _line(label, value, note=None):
    value = value if value is not None else "unknown"
    suffix = f"   ({note})" if note else ""
    return f"  {label:<22}: {value}{suffix}"


def print_report(info):
    print(f"Run statistics: {info['casepath']}  [{info['logfile']}]")
    print(_line("Originator", info["originator"], "assumes script runs as the run user/host"))
    print(_line("Run started", info["run_started"]))
    print(_line("Run finished", info["run_finished"]))

    wall = info["run_time_str"]
    if info["wall_seconds"] is not None:
        total = f"{int(round(info['wall_seconds']))} s total"
        wall = f"{wall}  ({total})" if wall else _fmt_duration(info["wall_seconds"])
    print(_line("Run time", wall))

    print(_line("Hermes-3 commit", info["hermes_commit"]))
    hg = info["hermes_git"]
    print(_line("Hermes-3 branch", hg["branch"],
                "from local git, not run dir" if hg["branch"] else "needs --git / local repo"))
    print(_line("Hermes-3 commit date", hg["date"],
                "from local git, not run dir" if hg["date"] else "needs --git / local repo"))

    print(_line("BOUT++ version", info["bout_version"]))
    print(_line("BOUT++ commit", info["bout_commit"]))
    bg = info["bout_git"]
    print(_line("BOUT++ branch", bg["branch"],
                "from local git, not run dir" if bg["branch"] else "needs --git / local repo"))
    print(_line("BOUT++ commit date", bg["date"],
                "from local git, not run dir" if bg["date"] else None))


# ------------------------------------------------------------
# PARSER
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise how/when/where a Hermes-3 run was produced, "
                    "using metadata from the simulation directory's BOUT.log file."
    )
    parser.add_argument("casepath", nargs="?", default=os.getcwd(),
                        help="Path to the simulation directory (default: cwd)")
    parser.add_argument("--git", dest="git", action="store_true", default=True,
                        help="Resolve branches/commit dates from local git repos (default: on)")
    parser.add_argument("--no-git", dest="git", action="store_false",
                        help="Use only information found in the simulation directory")
    args = parser.parse_args()

    info = runstats(args.casepath, use_git=args.git)
    print_report(info)
