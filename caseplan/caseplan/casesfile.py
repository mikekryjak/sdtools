"""The single ``cases.csv`` file: human-owned plan + tool-rendered status.

One CSV is both the thing you edit (case definitions) and the thing the tool
refreshes (status/evidence). The two never fight because ownership is *by
column*:

- **Human columns** (never overwritten by the tool): ``case, state, from,
  restart, hermes_build, changed_options, notes``.
- **Tool columns** (rewritten on every ``index``, never hand-edited):
  ``running, exists, final_t, runtime``.

``index`` merges on ``case``: your columns are kept verbatim, the tool columns
are refreshed from disk, and any run directory found on disk that is not yet in
the CSV is appended. Full provenance (commits, binary, mtimes, counts) lives in
each run's ``case.json``, not here.
"""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .boutinp import OptionChange, _split_lhs

# Human-owned state vocabulary. "running" is deliberately absent: liveness is a
# machine-detected column now, not a human judgement.
VALID_STATES = [
    "planned",
    "unfinished",
    "finished",
    "crashed",
    "stuck",
    "bad",
    "ignore",
    "unknown",
]

HUMAN_COLUMNS = [
    "case",
    "state",
    "from",
    "restart",
    "hermes_build",
    "changed_options",
    "notes",
]
TOOL_COLUMNS = ["running", "exists", "final_t", "runtime"]

# On-disk column order (reader-optimised; ownership is by name, not position).
COLUMNS = [
    "case",
    "state",
    "running",
    "exists",
    "from",
    "restart",
    "hermes_build",
    "changed_options",
    "notes",
    "final_t",
    "runtime",
]

_TRUE = {"yes", "true", "1", "y", "t"}


def _truthy(v: str) -> bool:
    return v.strip().lower() in _TRUE


def _yn(v: bool) -> str:
    return "yes" if v else "no"


@dataclass
class CaseRow:
    case: str
    # human-owned
    state: str = "unknown"
    from_: str = ""
    restart: str = ""
    hermes_build: str = ""
    changed_options: str = ""
    notes: str = ""
    # tool-owned (rendered)
    running: bool = False
    exists: bool = False
    final_t: str = ""
    runtime: str = ""

    # -- derived views of the human columns -------------------------------

    def restart_spec(self) -> tuple[str | None, str]:
        """Decode the ``restart`` cell into ``(restart_from, restart_mode)``.

        - empty / ``scratch`` -> ``(None, "scratch")``
        - bare ``restart`` / ``restart_append`` -> mode known, source not yet
          filled: ``(None, mode)``. ``casegen`` will demand a source before it
          regenerates such a case.
        - ``caseX`` -> restart from caseX; ``caseX:append`` -> restart_append
          from caseX.
        """
        s = self.restart.strip()
        if not s or s == "scratch":
            return None, "scratch"
        if s in ("restart", "restart_append"):
            return None, s
        if s.endswith(":append"):
            return s[: -len(":append")].strip(), "restart_append"
        return s, "restart"

    def option_changes(self) -> list[OptionChange]:
        return parse_changed_options(self.changed_options)


def parse_changed_options(text: str) -> list[OptionChange]:
    """Parse a ``;``-separated ``section:key=value`` list from one CSV cell.

    A non-empty item without ``=`` is an error, not silently ignored (mirrors
    the old ``Changes:`` safety rule).
    """
    changes: list[OptionChange] = []
    for item in text.split(";"):
        s = item.strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"changed_options item is not an assignment: {s!r}")
        lhs, value = s.split("=", 1)
        section, key = _split_lhs(lhs)
        if not key:
            raise ValueError(f"changed_options item has no option name: {s!r}")
        changes.append(OptionChange(section=section, key=key, value=value.strip(), raw=s))
    return changes


def changed_options_str(changes) -> str:
    return "; ".join(f"{c.dotted}={c.value}" for c in changes)


# ---------------------------------------------------------------------------
# Tool-column updates (only these fields are ever machine-written)
# ---------------------------------------------------------------------------

def apply_evidence(row: CaseRow, ev) -> None:
    row.exists = ev.exists
    row.running = ev.is_running
    row.final_t = "" if ev.final_t is None else f"{ev.final_t:.4g}"
    row.runtime = "" if ev.runtime is None else f"{ev.runtime:.4g}"


def clear_evidence(row: CaseRow) -> None:
    """For a row whose directory does not exist (a planned/queued case)."""
    row.exists = False
    row.running = False
    row.final_t = ""
    row.runtime = ""


# ---------------------------------------------------------------------------
# CSV read / write
# ---------------------------------------------------------------------------

def _row_from_dict(d: dict) -> CaseRow:
    def g(k: str) -> str:
        return (d.get(k) or "").strip()

    return CaseRow(
        case=g("case"),
        state=g("state") or "unknown",
        from_=g("from"),
        restart=g("restart"),
        hermes_build=g("hermes_build"),
        changed_options=g("changed_options"),
        notes=g("notes"),
        running=_truthy(g("running")),
        exists=_truthy(g("exists")),
        final_t=g("final_t"),
        runtime=g("runtime"),
    )


def _row_to_dict(r: CaseRow) -> dict:
    return {
        "case": r.case,
        "state": r.state,
        "running": _yn(r.running),
        "exists": _yn(r.exists),
        "from": r.from_,
        "restart": r.restart,
        "hermes_build": r.hermes_build,
        "changed_options": r.changed_options,
        "notes": r.notes,
        "final_t": r.final_t,
        "runtime": r.runtime,
    }


def read_cases(csv_path) -> list[CaseRow]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [_row_from_dict(d) for d in reader if (d.get("case") or "").strip()]


def write_cases(csv_path, rows: list[CaseRow]) -> Path:
    """Atomically write ``cases.csv``; back up any existing file to ``.bak``.

    Written via a temp file + ``os.replace`` so an interrupted write can never
    truncate the file you also hand-edit.
    """
    path = Path(csv_path)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    tmp = path.with_name(path.name + ".caseplan-tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(_row_to_dict(r))
    os.replace(tmp, path)
    return path
