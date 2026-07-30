"""Reading a results store back out.

The counterpart to extract.py. Extraction happens once and cannot be repeated
after the dumps are deleted; reading happens as often as the questions change.
Keeping them apart means analysis never touches a case directory, and so keeps
working on runs whose dumps are long gone.

Everything here reads the index and the bundles. Nothing reads a dump.
"""

import csv
import os

# Tables an extraction writes into each bundle, and what each one is a history
# of. Three different clocks, and mixing them silently is the easy mistake:
#   series      -- per output step, from the dump
#   steps       -- per output step, from BOUT.log.0
#   snes_steps  -- per INTERNAL solver step, so usually fewer rows than outputs
#                  once the timestep grows past the output interval
#   events      -- once per run, the log_view cost breakdown
TABLES = ("series", "steps", "snes_steps", "events")


class Store:
    """A results store: one index, many bundles."""

    def __init__(self, root, index_name="index.tsv"):
        self.root = root
        self.index_path = os.path.join(root, index_name)

    def rows(self, **filters):
        """
        Index rows, optionally filtered by exact column match.

        `state="recorded"` is the usual one. Values are strings, as stored —
        conversion is the caller's business, because what counts as a number
        differs by column and an empty cell must stay empty rather than become
        a zero.
        """

        with open(self.index_path, newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f, delimiter="\t")]

        for key, value in filters.items():
            rows = [r for r in rows if r.get(key) == value]
        return rows

    def row_for_case(self, case_dir):
        """The recorded row for a case directory name, or None."""

        key = os.path.basename(os.path.normpath(case_dir))
        matches = [
            r
            for r in self.rows()
            if os.path.basename(os.path.normpath(r.get("case_dir", ""))) == key
            and r.get("test_id")
        ]
        return matches[-1] if matches else None

    def bundle(self, test_id):
        return os.path.join(self.root, "runs", test_id)

    def table(self, test_id, name):
        """
        One bundle table as a dataframe, or None if this run has no such table.

        None is a real answer, not a failure: CVODE runs have no snes_steps, and
        a run launched without capturing stdout has no events.
        """

        import pandas as pd

        path = os.path.join(self.bundle(test_id), f"{name}.tsv")
        if not os.path.exists(path):
            return None
        index_col = 0 if name == "events" else None
        return pd.read_csv(path, sep="\t", index_col=index_col)

    def tables(self, test_id):
        """Every table a bundle holds, keyed by name, missing ones omitted."""

        found = {n: self.table(test_id, n) for n in TABLES}
        return {n: t for n, t in found.items() if t is not None}


def number(row, column, default=float("nan")):
    """A numeric cell, with empty meaning unknown rather than zero."""

    value = (row.get(column) or "").strip()
    try:
        return float(value)
    except ValueError:
        return default
