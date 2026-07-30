"""caseplan init: bootstrap ``cases.csv`` for an existing study directory.

Conservative by design. It fills the human columns it can infer (``from`` and ``changed_options`` via a
chosen/central base, ``restart`` mode from the case-name suffix) and the tool
columns from evidence, then leaves the file for you to review and own. It
refuses to overwrite an existing ``cases.csv`` unless ``--force`` is given.
"""

from __future__ import annotations

from pathlib import Path

from . import casesfile
from . import filesystem as fs
from . import index as index_mod
from .boutinp import read_bout_options


def _diff_options(base: dict[str, str], case: dict[str, str]) -> list[tuple[str, str]]:
    """Options where ``case`` differs from ``base`` (added or changed)."""
    return [(k, v) for k, v in case.items() if base.get(k) != v]


def _choose_base(cases: dict[str, dict[str, str]], explicit: str | None) -> str | None:
    if explicit and explicit in cases:
        return explicit
    if not cases:
        return None
    # Heuristic: the case whose options are most "central" (smallest total diff).
    best, best_score = None, None
    for cand in cases:
        score = sum(len(_diff_options(cases[cand], cases[other])) for other in cases)
        if best_score is None or score < best_score:
            best, best_score = cand, score
    return best


def _infer_restart(name: str) -> str:
    """Read the run mode from the case-name suffix (scratch/restart/append).

    Only the mode is inferable from a name; the restart *source* is not, so a
    bare mode word is written for you to complete if you regenerate the case.
    """
    low = name.lower()
    if "restart_append" in low or "append" in low:
        return "restart_append"
    if "restart" in low:
        return "restart"
    if "scratch" in low:
        return "scratch"
    return ""


class InitError(Exception):
    pass


def init_study(
    study_path,
    base: str | None = None,
    runs_dir: str = "runs",
    legacy_flat: bool = False,
    smart_diff: bool = True,
    force: bool = False,
) -> Path:
    study_path = Path(study_path)
    csv_path = study_path / "cases.csv"
    if csv_path.exists() and not force:
        raise InitError(
            f"{csv_path} already exists (human-owned). Refusing to overwrite. "
            f"Use --force to regenerate, or run `caseplan index` to refresh status."
        )

    run_dirs = fs.discover_run_dirs(study_path, runs_dir, legacy_flat)
    case_opts: dict[str, dict[str, str]] = {}
    for d in run_dirs:
        inp = d / "BOUT.inp"
        if inp.exists():
            case_opts[d.name] = read_bout_options(inp)

    base_name = _choose_base(case_opts, base) if smart_diff else None
    running_dirs = index_mod.live_bout_dirs()

    rows: list[casesfile.CaseRow] = []
    for d in run_dirs:
        name = d.name
        opts = case_opts.get(name, {})
        from_ = base_name if (base_name and name != base_name) else ""
        changed = ""
        if smart_diff and base_name and name != base_name:
            diffs = _diff_options(case_opts.get(base_name, {}), opts)
            changed = "; ".join(f"{k}={v}" for k, v in diffs)
        row = casesfile.CaseRow(
            case=name,
            state="unknown",
            from_=from_,
            restart=_infer_restart(name),
            changed_options=changed,
        )
        ev = index_mod.collect_evidence(d, None, running_dirs)
        index_mod.write_evidence(d, ev)
        casesfile.apply_evidence(row, ev)
        rows.append(row)

    return casesfile.write_cases(csv_path, rows)
