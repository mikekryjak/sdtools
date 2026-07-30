"""casegen: create or update the cases described by ``cases.csv``.

Paranoid by default. The safe path is to create new cases, never to mutate
existing expensive results. Existing output directories are refused unless an
explicit, deliberately-ugly flag is supplied.

Case definition comes entirely from a row's human columns: ``from`` (input
source), ``restart`` (``caseX`` / ``caseX:append``), and ``changed_options``.
The grid is a ``BOUT.inp`` setting, so it is not managed as a separate column.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import casesfile
from . import filesystem as fs
from .boutinp import apply_option_changes, format_line_change
from .casesfile import CaseRow


class GenerationError(Exception):
    pass


@dataclass
class GenOptions:
    dry_run: bool = False
    case: str | None = None
    update_inputs_only: bool = False
    runs_dir: str = "runs"
    legacy_flat: bool = False
    force_existing: bool = False
    allow_output_dir: bool = False
    overwrite_handmade: bool = False


@dataclass
class GenResult:
    name: str
    target: Path
    copies: list = field(default_factory=list)
    line_changes: list = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    skipped: bool = False


def _resolve_source(root: Path, ref: str, runs_dir: str) -> Path:
    """Find a directory to copy from, given a ``from``/``restart`` case ref."""
    if not ref:
        raise GenerationError("no source case given")
    candidates = [root / runs_dir / ref, root / ref, Path(ref)]
    for c in candidates:
        if (c / "BOUT.inp").exists():
            return c
    raise GenerationError(f"cannot resolve source {ref!r} (no BOUT.inp found)")


def _target_dir(root: Path, name: str, opts: GenOptions) -> Path:
    if opts.legacy_flat:
        return root / name
    return root / opts.runs_dir / name


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _check_safety(target: Path, opts: GenOptions) -> None:
    if not target.exists():
        return

    case_json = target / "case.json"
    has_gen = case_json.exists() and "generation" in _read_json(case_json)

    if not has_gen:
        if not opts.overwrite_handmade:
            raise GenerationError(
                f"{target} exists and has no case.json generation block "
                f"(hand-made). Refusing. Use --overwrite-handmade to force."
            )
        return

    if fs.is_protected_dir(target):
        if not opts.allow_output_dir:
            raise GenerationError(
                f"{target} contains run outputs. Refusing. "
                f"Use --allow-output-dir to force (rare)."
            )
        return

    if not (opts.force_existing or opts.update_inputs_only):
        raise GenerationError(
            f"{target} is an existing generated case. "
            f"Use --update-inputs-only or --force-existing."
        )


def _write_generation(target: Path, root: Path, row: CaseRow, source: Path,
                      restart_from: str | None, restart_mode: str,
                      copies, line_changes) -> None:
    from . import __version__

    case_json = target / "case.json"
    data = _read_json(case_json) if case_json.exists() else {}
    data["generation"] = {
        "case": row.case,
        "from": row.from_,
        "source": str(source),
        "restart": row.restart,
        "restart_from": restart_from,
        "restart_mode": restart_mode,
        "hermes_build": row.hermes_build,
        "changed_options": row.changed_options,
        "copied_files": [a.dst.name for a in copies if a.kind in {"input", "script", "extra"}],
        "copied_restarts": [a.dst.name for a in copies if a.kind == "restart"],
        "applied_changes": [
            {"option": lc.change.dotted, "value": lc.change.value, "action": lc.action}
            for lc in line_changes
        ],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool_version": __version__,
    }
    tmp = case_json.with_name("case.json.caseplan-tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, case_json)


def generate_case(root: Path, row: CaseRow, opts: GenOptions) -> GenResult:
    source = _resolve_source(root, row.from_, opts.runs_dir)
    target = _target_dir(root, row.case, opts)
    result = GenResult(name=row.case, target=target)

    _check_safety(target, opts)

    changes = row.option_changes()
    restart_from, restart_mode = row.restart_spec()

    copies = fs.plan_input_copies(source, target)

    if restart_mode in {"restart", "restart_append"}:
        if not restart_from:
            raise GenerationError(f"{row.case}: restart={row.restart!r} needs a source case")
        restart_src = _resolve_source(root, restart_from, opts.runs_dir)
        copies += fs.plan_restart_copies(restart_src, target)
    result.copies = copies

    src_inp = source / "BOUT.inp"

    if opts.dry_run:
        result.messages.append(f"[dry-run] source: {source}")
        result.messages.append(f"[dry-run] target: {target}")
        for a in copies:
            result.messages.append(f"  copy {a.kind}: {a.src.name} -> {a.dst}")
        line_changes = apply_option_changes(src_inp, changes, dry_run=True)
        result.line_changes = line_changes
        for lc in line_changes:
            result.messages.append(format_line_change(lc))
        return result

    target.mkdir(parents=True, exist_ok=True)
    fs.execute_copies(copies)

    target_inp = target / "BOUT.inp"
    line_changes = apply_option_changes(target_inp, changes, dry_run=False)
    result.line_changes = line_changes

    if opts.allow_output_dir or opts.overwrite_handmade:
        flag = "--allow-output-dir" if opts.allow_output_dir else "--overwrite-handmade"
        result.messages.append(f"WARNING: wrote into protected/hand-made dir via {flag}")

    _write_generation(target, root, row, source, restart_from, restart_mode, copies, line_changes)
    result.messages.append(f"generated {target}")
    return result


def casegen(cases_csv, opts: GenOptions) -> list[GenResult]:
    csv_path = Path(cases_csv)
    root = csv_path.parent
    rows = casesfile.read_cases(csv_path)

    if opts.case:
        rows = [r for r in rows if r.case == opts.case]
        if not rows:
            raise GenerationError(f"No case named {opts.case!r} in {csv_path}")

    results = []
    for row in rows:
        _, restart_mode = row.restart_spec()
        if not row.changed_options.strip() and restart_mode == "scratch":
            # Nothing to build: an exact scratch copy of the parent. Skip.
            results.append(GenResult(
                name=row.case, target=_target_dir(root, row.case, opts),
                skipped=True,
                messages=["skipped: no changed_options and no restart"],
            ))
            continue
        results.append(generate_case(root, row, opts))
    return results
