"""The results index: one tab-separated row per test.

The column meanings, the row lifecycle and the reasoning behind them live in the
results store's `schema.md`. This module owns only the mechanics, and one rule
that matters more than the rest: the index is written by hand as well as by
code. A row is created by a person before the run, carrying intent; the
extractor fills in what it measured. So nothing here may reorder rows, rewrite a
cell that already has a value, or touch a row it did not open.
"""

import csv
import os

STATE_PLANNED = "planned"
STATE_RECORDED = "recorded"
STATE_UNPLANNED = "unplanned"
STATE_CANCELLED = "cancelled"

# Declared by hand before the run. Everything else is measured.
INTENT_COLUMNS = ["case_dir", "test", "recipe", "varied", "epoch", "note"]

# Canonical column order. Meaning first -- what was tried, what happened, how
# much it cost -- with provenance and bookkeeping trailing, so nobody has to
# scroll past twenty columns to find the point.
INDEX_COLUMNS = [
    # what was tried
    "test_id",
    "case_dir",
    "state",
    "test",
    "recipe",
    "varied",
    # what happened
    "outcome",
    "wall_s",
    "sim_time_ms",
    "verdict",
    # what it cost
    "ncalls",
    "nl_its",
    "lin_its",
    "ms_per_24h",
    "solver_fails",
    # where the time went
    "t_jac_frac",
    "t_pcsetup_frac",
    "t_ksp_frac",
    "t_func_frac",
    "n_jac_builds",
    # convergence
    "resid_final",
    "resid_drop",
    "resid_per_rhs",
    # correctness
    "ne_sep",
    "te_sep",
    "ne_target_max",
    "te_target_max",
    "max_dev",
    "reference_id",
    # what it ran as
    "diffs",
    "epoch",
    "hermes_commit",
    "hermes_branch",
    "bout_version",
    "bout_commit",
    "petsc_version",
    "grid",
    "seed",
    "run_id",
    "limiter",
    "conduction_method",
    "check_level",
    "cores",
    "decomposition",
    "slot",
    "concurrency",
    "machine",
    # bookkeeping
    "run_started",
    "recorded_at",
    "originator",
    "note",
]


class IndexProblem(Exception):
    """Something about the index needs a person, not a guess."""


def read_index(path):
    """Rows as a list of dicts, and the column order actually in the file.

    A missing file is not an error: it yields no rows and the canonical column
    order, so the first extraction creates it.
    """

    if not os.path.exists(path):
        return [], list(INDEX_COLUMNS)

    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [dict(row) for row in reader]
        columns = list(reader.fieldnames or INDEX_COLUMNS)

    return rows, columns


def write_index(path, rows, columns):
    """Write the index back, atomically.

    Via a temporary file and a rename, because this file is one a person may
    have open in a spreadsheet: a half-written index is worse than none.
    """

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    os.replace(tmp, path)


def case_key(case_dir):
    """The join key between a planned row and a finished directory.

    The directory's own name, not its path: the index is written by hand and
    may carry either, and a case is identified by which directory it ran in.
    """

    return os.path.basename(os.path.normpath(case_dir))


def find_open_row(rows, case_dir):
    """
    Position of the one `planned` row for this case directory, or None.

    Two open rows for one directory is an error rather than a guess. A directory
    is reused across tests, so guessing which row a finished run belongs to
    would silently attach results to the wrong experiment.
    """

    key = case_key(case_dir)
    matches = [
        i
        for i, row in enumerate(rows)
        if case_key(row.get("case_dir", "")) == key
        and row.get("state", "") == STATE_PLANNED
    ]

    if len(matches) > 1:
        raise IndexProblem(
            f"{len(matches)} rows are open for {key} (rows"
            f" {', '.join(str(i + 2) for i in matches)} of the file). At most one"
            " row per case_dir may be `planned` at a time — close or cancel the"
            " stale one."
        )

    return matches[0] if matches else None


def fill_row(row, measured, columns):
    """
    Write measured values into a row's empty cells, and report the rest.

    Returns (filled, conflicts, unknown). A cell that already has a value is
    never overwritten: a declared value is somebody's statement of intent, and
    where measurement disagrees with it that disagreement is the finding — a
    recipe drifted, a stale input file was used, the wrong case was launched.
    """

    filled, conflicts, unknown = [], {}, []

    for key, value in measured.items():
        if value is None or value == "":
            continue
        text = _as_text(value)
        if key not in columns:
            unknown.append(key)
            continue
        existing = (row.get(key) or "").strip()
        if not existing:
            row[key] = text
            filled.append(key)
        elif existing != text:
            conflicts[key] = (existing, text)

    return filled, conflicts, unknown


def recompute_concurrency(rows):
    """
    Fill `concurrency` on every row that can support it, and say how many changed.

    This cannot be done when a run is extracted, because a run that overlapped it
    may not be recorded until later. So it is a pass over the whole index,
    re-run after each extraction.

    Counted as the number of runs sharing the machine, including this one: 1
    means it ran alone, 3 means three at once. Any temporal overlap counts, so a
    run that only briefly coincided with another still reads as 2 — the field
    exists to say "this timing may be contaminated", and a brief overlap is
    exactly the case worth looking at rather than hiding.

    This is what F3 blames for every unattributable comparison so far: slot was
    recorded as free text and loading was never recorded at all.
    """

    import datetime

    intervals = []
    for row in rows:
        start, wall = row.get("run_started", ""), row.get("wall_s", "")
        try:
            t0 = datetime.datetime.strptime(start.strip(), "%a %b %d %H:%M:%S %Y")
            span = float(wall)
        except (ValueError, AttributeError):
            intervals.append(None)
            continue
        intervals.append((t0, t0 + datetime.timedelta(seconds=span)))

    changed = 0
    for i, own in enumerate(intervals):
        if own is None:
            continue
        overlaps = sum(
            1
            for j, other in enumerate(intervals)
            if other is not None and i != j and other[0] < own[1] and own[0] < other[1]
        )
        value = str(1 + overlaps)
        if rows[i].get("concurrency", "") != value:
            rows[i]["concurrency"] = value
            changed += 1

    return changed


def _as_text(value):
    """Index cells are text. Floats keep full precision so a value read back
    from the index equals the one that was measured."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)
