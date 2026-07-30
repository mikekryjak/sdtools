"""case-index: refresh the tool-owned columns of ``cases.csv``.

Reads each run directory, refreshes the ``evidence`` block of its ``case.json``
(full provenance), and merges the four tool columns (``running, exists,
final_t, runtime``) back into ``cases.csv`` without touching the human columns.

State is human-owned and lives only in the ``state`` column; it is never
inferred from logs. Logs/PID files populate evidence only.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import casesfile
from . import filesystem as fs
from .boutinp import read_bout_options

_SIMTIME_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b")
_ERROR_RE = re.compile(
    r"(\berror\b|\bfatal\b|\babort\w*|sigfpe|sigsegv|segmentation fault|nan detected|not converged)",
    re.I,
)
# Lines echoing resolved options ("Option foo:error_bar = 1") are not errors.
_OPTION_ECHO_RE = re.compile(r"^\s*Option\s")
# Benign BOUT++ startup lines that contain "error" but report no failure.
_BENIGN_ERROR_RE = re.compile(r"error checking (disabled|enabled)", re.I)


# ---------------------------------------------------------------------------
# Running detection: match a live process's ``-d <datadir>`` against a case dir
# ---------------------------------------------------------------------------

def _extract_d_arg(args: list[str]) -> str | None:
    """Return the value of a BOUT ``-d`` data-directory argument, if present."""
    for i, a in enumerate(args):
        if a == "-d":
            return args[i + 1] if i + 1 < len(args) else None
        if a.startswith("-d=") and len(a) > 3:
            return a[3:]
        if a.startswith("-d") and len(a) > 2 and a != "-d":
            return a[2:]
    return None


def _resolve_dir(d_arg: str, cwd: str | None) -> Path | None:
    p = Path(d_arg)
    try:
        if p.is_absolute():
            return p.resolve()
        if cwd:
            return (Path(cwd) / p).resolve()
        return p.resolve()
    except OSError:
        return None


def live_bout_dirs() -> set[Path]:
    """Resolved data directories of every live local process carrying ``-d DIR``.

    A case is running iff its resolved directory is in this set. Matching the
    exact ``-d`` argument (not a substring) is essential: ``st40fllrb4`` is a
    substring of ``st40fllrb4fb``, so loose matching would cross-report runs.
    """
    dirs: set[Path] = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return dirs
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        args = [a.decode(errors="replace") for a in raw.split(b"\x00") if a]
        d_arg = _extract_d_arg(args)
        if d_arg is None:
            continue
        try:
            cwd = os.readlink(entry / "cwd")
        except OSError:
            cwd = None
        target = _resolve_dir(d_arg, cwd)
        if target is not None:
            dirs.add(target)
    return dirs


# ---------------------------------------------------------------------------
# Evidence extraction (full detail; stored in case.json)
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    exists: bool = False
    final_t: float | None = None
    runtime: float | None = None
    planned_hermes_build: str | None = None
    run_binary: str | None = None
    hermes_commit: str | None = None
    bout_version: str | None = None
    bout_commit: str | None = None
    last_log_mtime: str | None = None
    last_output_mtime: str | None = None
    error_hint: str | None = None
    progress_hint: str | None = None
    is_running: bool = False
    running_evidence: str | None = None
    n_logs: int = 0
    n_dumps: int = 0
    n_restarts: int = 0
    has_squash: bool = False
    indexed_at: str = ""
    source_mtimes: dict[str, float] = field(default_factory=dict)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _pick_log(run_dir: Path) -> Path | None:
    log0 = run_dir / "BOUT.log.0"
    if log0.exists():
        return log0
    logs = sorted(run_dir.glob("BOUT.log.*"))
    return logs[0] if logs else None


def _parse_log(log_path: Path, ev: Evidence) -> None:
    final_t = None
    wall_sum = 0.0
    last_error = None
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return
    for line in text.splitlines():
        if ev.bout_version is None:
            m = re.search(r"BOUT\+\+ version\s+(\S+)", line)
            if m:
                ev.bout_version = m.group(1)
        if ev.bout_commit is None:
            m = re.match(r"\s*Revision:\s*(\S+)", line)
            if m:
                ev.bout_commit = m.group(1)
        if ev.hermes_commit is None:
            m = re.search(r"Git Version of Hermes:\s*(\S+)", line)
            if m:
                ev.hermes_commit = m.group(1)
        if ev.run_binary is None:
            m = re.search(r"Command line options for this run\s*:\s*(\S+)", line)
            if m:
                ev.run_binary = m.group(1)
        sm = _SIMTIME_RE.match(line)
        if sm and "|" not in line:
            final_t = float(sm.group(1))
            wall_sum += float(sm.group(3))
        if (
            not _OPTION_ECHO_RE.match(line)
            and not _BENIGN_ERROR_RE.search(line)
            and _ERROR_RE.search(line)
        ):
            last_error = line.strip()

    if final_t is not None:
        ev.final_t = final_t
    if wall_sum > 0:
        ev.runtime = wall_sum
    if last_error:
        ev.error_hint = last_error[:200]


def collect_evidence(
    run_dir,
    planned_hermes_build: str | None = None,
    running_dirs: set[Path] | None = None,
) -> Evidence:
    run_dir = Path(run_dir)
    ev = Evidence(indexed_at=_iso(datetime.now().timestamp()))
    ev.exists = run_dir.is_dir()
    ev.planned_hermes_build = planned_hermes_build
    if not ev.exists:
        return ev

    counts = fs.count_outputs(run_dir)
    ev.n_logs = counts.n_logs
    ev.n_dumps = counts.n_dumps
    ev.n_restarts = counts.n_restarts
    ev.has_squash = counts.has_squash

    log = _pick_log(run_dir)
    if log is not None:
        _parse_log(log, ev)

    log_mtimes = [p.stat().st_mtime for p in run_dir.glob("BOUT.log.*")]
    out_mtimes = [
        p.stat().st_mtime
        for p in list(run_dir.glob("BOUT.dmp.*.nc"))
        + list(run_dir.glob("BOUT.restart.*.nc"))
        + list(run_dir.glob("BOUT.squash.nc"))
    ]
    if log_mtimes:
        last_log = max(log_mtimes)
        ev.last_log_mtime = _iso(last_log)
        ev.source_mtimes["log"] = last_log
    if out_mtimes:
        last_out = max(out_mtimes)
        ev.last_output_mtime = _iso(last_out)
        ev.source_mtimes["output"] = last_out

    if running_dirs is None:
        running_dirs = live_bout_dirs()
    if run_dir.resolve() in running_dirs:
        ev.is_running = True
        ev.running_evidence = "live process with -d matching this directory"
    else:
        newest = max(log_mtimes + out_mtimes, default=0.0)
        if newest and (datetime.now().timestamp() - newest) < 300:
            ev.running_evidence = f"recent activity {ev.last_log_mtime or ev.last_output_mtime}"

    return ev


# ---------------------------------------------------------------------------
# case.json (preserve generation, replace evidence)
# ---------------------------------------------------------------------------

def _evidence_to_dict(ev: Evidence) -> dict:
    return {
        "exists": ev.exists,
        "final_t": ev.final_t,
        "runtime": ev.runtime,
        "planned_hermes_build": ev.planned_hermes_build,
        "run_binary": ev.run_binary,
        "hermes_commit": ev.hermes_commit,
        "bout_version": ev.bout_version,
        "bout_commit": ev.bout_commit,
        "last_log_mtime": ev.last_log_mtime,
        "last_output_mtime": ev.last_output_mtime,
        "error_hint": ev.error_hint,
        "progress_hint": ev.progress_hint,
        "is_running": ev.is_running,
        "running_evidence": ev.running_evidence,
        "n_logs": ev.n_logs,
        "n_dumps": ev.n_dumps,
        "n_restarts": ev.n_restarts,
        "has_squash": ev.has_squash,
        "indexed_at": ev.indexed_at,
        "source_mtimes": ev.source_mtimes,
    }


def write_evidence(run_dir: Path, ev: Evidence) -> None:
    """Rewrite only the ``evidence`` block of ``case.json``; keep ``generation``."""
    case_json = Path(run_dir) / "case.json"
    data: dict = {}
    if case_json.exists():
        try:
            data = json.loads(case_json.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
    data["evidence"] = _evidence_to_dict(ev)
    case_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = case_json.with_name("case.json.caseplan-tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, case_json)


# ---------------------------------------------------------------------------
# Merge: refresh tool columns of cases.csv, append newly-found runs
# ---------------------------------------------------------------------------

def refresh(study_path, runs_dir: str = "runs", legacy_flat: bool = False) -> list[casesfile.CaseRow]:
    study_path = Path(study_path)
    csv_path = study_path / "cases.csv"
    rows = casesfile.read_cases(csv_path)
    by_name = {r.case: r for r in rows}

    run_dirs = fs.discover_run_dirs(study_path, runs_dir, legacy_flat)
    disk = {d.name: d for d in run_dirs}
    running_dirs = live_bout_dirs()

    for r in rows:
        d = disk.get(r.case)
        if d is not None:
            ev = collect_evidence(d, r.hermes_build or None, running_dirs)
            write_evidence(d, ev)
            casesfile.apply_evidence(r, ev)
        else:
            casesfile.clear_evidence(r)

    # Discovered runs not yet tracked: append so a run never silently vanishes.
    for name, d in disk.items():
        if name in by_name:
            continue
        r = casesfile.CaseRow(case=name)
        ev = collect_evidence(d, None, running_dirs)
        write_evidence(d, ev)
        casesfile.apply_evidence(r, ev)
        rows.append(r)

    casesfile.write_cases(csv_path, rows)
    return rows


# Backwards-compatible name used by the CLI.
def case_index(study_path, runs_dir: str = "runs", legacy_flat: bool = False) -> list[casesfile.CaseRow]:
    return refresh(study_path, runs_dir, legacy_flat)


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------

_TABLE_COLS = ["case", "state", "running", "exists", "final_t", "runtime", "from", "notes"]


def format_table(rows: list[casesfile.CaseRow]) -> str:
    def cell(r: casesfile.CaseRow, c: str) -> str:
        if c == "running":
            return "yes" if r.running else "no"
        if c == "exists":
            return "yes" if r.exists else "no"
        if c == "from":
            return r.from_
        return getattr(r, c)

    widths = {c: len(c) for c in _TABLE_COLS}
    for r in rows:
        for c in _TABLE_COLS:
            widths[c] = max(widths[c], len(str(cell(r, c))))
    widths["notes"] = min(widths["notes"], 40)

    def fmt(get) -> str:
        return "  ".join(str(get(c))[: widths[c]].ljust(widths[c]) for c in _TABLE_COLS)

    header = fmt(lambda c: c)
    sep = "  ".join("-" * widths[c] for c in _TABLE_COLS)
    body = "\n".join(fmt(lambda c, r=r: cell(r, c)) for r in rows)
    return f"{header}\n{sep}\n{body}"
