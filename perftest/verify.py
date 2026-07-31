"""Re-derive a stored record from its raw artefacts and compare (R23).

The point is narrow and worth stating, because it is easy to over-read this
check. It proves the extractor still produces what is on disk -- so a change
that silently alters a number shows up as a difference rather than as a quietly
different record. It does NOT prove the stored number was ever right: if a value
was wrong at first extraction and the code has not changed since, it re-derives
the same wrong value and reports a match. The guard-cell bug was exactly that
shape, which is why this compares bundle tables and not only index columns --
the contamination lived in the tables.

Runnable against any case directory that still exists. Once R8 deletes a case,
its row can no longer be checked this way, so the useful window is between
extraction and deletion.
"""

import os
import shutil
import tempfile

# Bookkeeping that legitimately differs on a re-run: when it was recorded, who
# ran the check, and the fields that describe intent rather than measurement.
IGNORED = {
    "recorded_at", "originator", "state", "note", "varied", "epoch",
    "conduction_method", "test_id", "concurrency",
}


class Difference:
    def __init__(self, where, key, stored, fresh):
        self.where, self.key, self.stored, self.fresh = where, key, stored, fresh

    def __str__(self):
        return f"{self.where}: {self.key}: stored {self.stored!r} -> {self.fresh!r}"


def _close(stored, fresh, rtol):
    """Equal as text, or as numbers within rtol. Text first, because most
    columns are strings and float() on a commit sha is a waste."""

    if str(stored) == str(fresh):
        return True
    try:
        a, b = float(stored), float(fresh)
    except (TypeError, ValueError):
        return False
    if a == b:
        return True
    return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))


def _compare_tables(stored_dir, fresh_dir, rtol):
    """Every parsed table in the bundle, cell by cell."""

    import pandas as pd

    out = []
    names = sorted({f for f in os.listdir(stored_dir) if f.endswith(".tsv")}
                   | {f for f in os.listdir(fresh_dir) if f.endswith(".tsv")})
    for name in names:
        stored_path = os.path.join(stored_dir, name)
        fresh_path = os.path.join(fresh_dir, name)
        if not os.path.exists(stored_path):
            out.append(Difference(name, "(table)", "absent", "present"))
            continue
        if not os.path.exists(fresh_path):
            out.append(Difference(name, "(table)", "present", "absent"))
            continue

        a = pd.read_csv(stored_path, sep="\t")
        b = pd.read_csv(fresh_path, sep="\t")
        if list(a.columns) != list(b.columns):
            out.append(Difference(name, "(columns)", list(a.columns),
                                  list(b.columns)))
            continue
        if len(a) != len(b):
            out.append(Difference(name, "(rows)", len(a), len(b)))
            continue

        for column in a.columns:
            for i, (x, y) in enumerate(zip(a[column], b[column])):
                # NaN is a legitimate value here: the sparse residual tables
                # leave a region blank rather than writing a zero it did not
                # measure, so NaN on both sides is agreement.
                if x != x and y != y:
                    continue
                if not _close(x, y, rtol):
                    out.append(Difference(name, f"{column}[{i}]", x, y))
                    break  # one report per column is enough to investigate
    return out


def rederive(case_dir, store_dir, row, rtol=1e-9, keep=False):
    """
    Re-extract `case_dir` and compare against the stored row and bundle.

    row : the stored index row for this case, as a dict.

    Returns a list of Difference. Empty means the record on disk is exactly what
    today's code produces from today's artefacts.
    """

    from .extract import extract_case

    scratch = tempfile.mkdtemp(prefix="perftest-verify-")
    try:
        report = extract_case(case_dir, scratch)
        diffs = []

        for key, value in report.record.items():
            if key in IGNORED:
                continue
            stored = (row.get(key) or "").strip()
            if not stored:
                continue
            if not _close(stored, value, rtol):
                diffs.append(Difference("index", key, stored, value))

        # A row migrated from the legacy CSV has no bundle and never did, so a
        # missing one is "nothing to check", not a disagreement. Only a row the
        # extractor itself wrote is expected to have tables behind it.
        test_id = row.get("test_id") or report.test_id
        stored_bundle = os.path.join(store_dir, "runs", test_id or "")
        if report.bundle and os.path.isdir(stored_bundle):
            diffs += _compare_tables(stored_bundle, report.bundle, rtol)
        elif test_id and row.get("state", "") in ("recorded", "unplanned"):
            diffs.append(Difference("bundle", test_id, "expected", "not found"))

        return diffs
    finally:
        if not keep:
            shutil.rmtree(scratch, ignore_errors=True)
