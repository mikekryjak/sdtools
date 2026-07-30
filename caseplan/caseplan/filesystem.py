"""Filesystem helpers: output detection, copy policy, run discovery.

The copy policy is paranoid by default: never copy run products, PID/scheduler
markers, or monitor plots out of a source case, because inherited markers can
make ``is_running`` evidence wrong.
"""

from __future__ import annotations

import fnmatch
import shutil
from dataclasses import dataclass
from pathlib import Path

# Files whose presence makes a directory a real (protected) run.
PROTECTED_GLOBS = [
    "BOUT.log.*",
    "BOUT.dmp.*.nc",
    "BOUT.restart.*.nc",
    "BOUT.squash.nc",
]

# Never copied out of a source case.
EXCLUDE_GLOBS = [
    "BOUT.log.*",
    "BOUT.dmp.*.nc",
    "BOUT.restart.*.nc",
    "BOUT.squash.nc",
    "BOUT.settings",
    "*.nc",  # grids are copied separately, by explicit policy
    "*.png",  # monitor plots unless explicitly requested
    ".BOUT.pid.*",
    "*.pid",
    "*.lock",
    "case.json",
    ".status",
    "*kate-swp*",
    "*~",
]

# Helper/submit scripts worth carrying into a new run by default.
SCRIPT_GLOBS = ["*.sh", "submit*", "job*", "*.pbs", "*.slurm", "*.job"]


def _matches(name: str, globs) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in globs)


def is_excluded(name: str) -> bool:
    return _matches(name, EXCLUDE_GLOBS)


def is_protected_dir(d) -> bool:
    """True if the directory contains run products that must not be clobbered."""
    d = Path(d)
    if not d.is_dir():
        return False
    for child in d.iterdir():
        if child.is_file() and _matches(child.name, PROTECTED_GLOBS):
            return True
    return False


@dataclass
class OutputCounts:
    n_logs: int = 0
    n_dumps: int = 0
    n_restarts: int = 0
    has_squash: bool = False
    n_monitors: int = 0


def count_outputs(d) -> OutputCounts:
    d = Path(d)
    c = OutputCounts()
    if not d.is_dir():
        return c
    for child in d.iterdir():
        n = child.name
        if fnmatch.fnmatch(n, "BOUT.log.*"):
            c.n_logs += 1
        elif fnmatch.fnmatch(n, "BOUT.dmp.*.nc"):
            c.n_dumps += 1
        elif fnmatch.fnmatch(n, "BOUT.restart.*.nc"):
            c.n_restarts += 1
        elif n == "BOUT.squash.nc":
            c.has_squash = True
        elif fnmatch.fnmatch(n, "monitor*.png"):
            c.n_monitors += 1
    return c


def looks_like_case(d) -> bool:
    """A directory is a case if it has inputs or outputs we recognise."""
    d = Path(d)
    if not d.is_dir():
        return False
    if (d / "BOUT.inp").exists() or (d / "BOUT.settings").exists():
        return True
    return any(fnmatch.fnmatch(c.name, "BOUT.dmp.*.nc") for c in d.iterdir())


def discover_run_dirs(study_path, runs_dir: str = "runs", legacy_flat: bool = False) -> list[Path]:
    """Find case directories under ``runs/`` and/or as direct study children."""
    study_path = Path(study_path)
    found: list[Path] = []
    seen: set[Path] = set()

    runs_root = study_path / runs_dir
    if runs_root.is_dir():
        for child in sorted(runs_root.iterdir()):
            if child.is_dir() and looks_like_case(child) and child not in seen:
                found.append(child)
                seen.add(child)

    if legacy_flat or not runs_root.is_dir():
        for child in sorted(study_path.iterdir()):
            if child.name in {runs_dir, "templates", "grids"}:
                continue
            if child.is_dir() and looks_like_case(child) and child not in seen:
                found.append(child)
                seen.add(child)

    return found


@dataclass
class CopyAction:
    src: Path
    dst: Path
    kind: str  # "input" | "script" | "extra" | "grid" | "restart"


def plan_input_copies(src_dir, dst_dir, extra: list[str] | None = None) -> list[CopyAction]:
    """List the copy actions for ordinary inputs (no outputs, no restarts)."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    actions: list[CopyAction] = []
    extra = extra or []

    inp = src_dir / "BOUT.inp"
    if inp.exists():
        actions.append(CopyAction(inp, dst_dir / "BOUT.inp", "input"))

    for child in sorted(src_dir.iterdir()):
        if not child.is_file() or child.name == "BOUT.inp":
            continue
        if is_excluded(child.name):
            continue
        if _matches(child.name, SCRIPT_GLOBS):
            actions.append(CopyAction(child, dst_dir / child.name, "script"))

    for name in extra:
        cand = src_dir / name
        if cand.exists():
            actions.append(CopyAction(cand, dst_dir / cand.name, "extra"))

    return actions


def plan_restart_copies(src_dir, dst_dir) -> list[CopyAction]:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    actions = []
    for child in sorted(src_dir.glob("BOUT.restart.*.nc")):
        actions.append(CopyAction(child, dst_dir / child.name, "restart"))
    return actions


def execute_copies(actions: list[CopyAction]) -> None:
    for a in actions:
        a.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.src, a.dst)
