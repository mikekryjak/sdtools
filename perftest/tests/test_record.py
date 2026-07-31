"""Tests for the paths between a finished run and its recorded row (R23).

`test_extract.py` covers the field reductions. This covers everything else the
extractor leans on: reading what a run actually printed, diffing it against the
recipe it was supposed to be, and writing the result into an index a person also
edits by hand.

These are the quiet failure modes. A reduction that includes guard cells reports
an obviously strange number; a row lifecycle that silently overwrites a declared
value, or a log parser that drops half a table because the format shifted,
produces a record that looks entirely reasonable and is wrong. Hundreds of rows
are machine-produced, so anything that fails quietly fails at scale.
"""

import pathlib
import sys

import pytest

from hermes3 import logparse as lp
from perftest import index as idx
from perftest import recipe


# =============================================================================
# Fixtures: the smallest files that exercise each parser
# =============================================================================
LOGVIEW = """\
Using PETSc Release Version 3.23.3, May 30, 2025

Event                Count      Time (sec)     Flop                  --- Global ---  --- Stage ----  Total
                   Max Ratio  Max     Ratio   Max  Ratio  Mess   AvgLen  Reduct  %T %F %M %L %R  %T %F %M %L %R Mflop/s
------------------------------------------------------------------------------------------------------------------------

--- Event Stage 0: Main Stage

BuildTwoSided        661 1.0 2.6895e-02 3.3 0.00e+00 0.0 1.3e+04 4.0e+00 6.6e+02  0  0 14  0 15   0  0 14  0 15     0
Total BOUT++           1 1.0 1.5896e+02 1.0 3.05e+09 1.5 9.1e+04 4.6e+03 4.5e+03 100 100 100 100 100 100 100 100 100 100   163
SNESJacobianEval     329 1.0 1.2321e+02 1.0 8.97e+08 2.0 0.0e+00 0.0e+00 2.0e+00 78 25  0  0  0  78 25  0  0  0   275
------------------------------------------------------------------------------------------------------------------------

Object Type          Creations   Destructions. Reports information only for process 0.

--- Event Stage 0: Main Stage

           Container     2              0
"""

BOUT_LOG = """\
BOUT++ version 5.2.1
Revision: 2384555b6ad9763cddc3e99a99ed4c4c1037faa2
Git Version of Hermes: a1c3ba381f87686d5a6ee8bf52226886ece6bb1d
	Runtime error checking enabled, level 2
Run started at  : Wed Jul 29 17:43:13 2026

Sim Time  |  RHS evals  | Wall Time |  Calc    Inv   Comm    I/O   SOLVER

9.579e+05          1       7.55e-02    -1.8    0.0    5.9  190.7  -94.7
Time: 957880.1, timestep: 0.1, nl iter: 2, lin iter: 3, reason: 3
1.042e+06          7       1.06e-01     2.6    0.0    0.2   96.6    0.6
Time: 1038172.9, timestep: 2500.2, nl iter: 7, lin iter: 9, reason: 4, SNES failures: 2
Run finished at  : Wed Jul 29 17:45:52 2026
Run time : 2 m 39 s
"""

RECIPE_TEXT = """\
# A-RECIPE
[solver]
type = snes
atol = 1e-7
lag_jacobian = 1

[petsc]
log_view
pc_type = lu
"""


@pytest.fixture
def case(tmp_path):
    """A case directory carrying the text files the extractor reads."""
    (tmp_path / "BOUT.log.0").write_text(BOUT_LOG)
    (tmp_path / "BOUT.log.console").write_text(LOGVIEW)
    return tmp_path


def _rows(*specs):
    """Index rows from (case_dir, state) pairs, every other column empty."""
    out = []
    for case_dir, state in specs:
        row = {c: "" for c in idx.INDEX_COLUMNS}
        row["case_dir"], row["state"] = case_dir, state
        out.append(row)
    return out


# =============================================================================
# log_view: the report is data, and a format change must not be silent
# =============================================================================
def test_event_names_containing_spaces_survive(case):
    """`Total BOUT++` is one event, not a name and a stray column. Splitting on
    whitespace from the left loses it; this is why rows split from the right."""
    df = lp.parse_petsc_logview(str(case / "BOUT.log.console"))
    assert "Total BOUT++" in df.index
    assert df["time_max"]["Total BOUT++"] == pytest.approx(158.96)


def test_the_object_table_is_not_swallowed(case):
    """The report has a second `Event Stage` heading over a table of object
    creations. Reading past the timing table into it is what the old parser did
    before it hit an IndexError."""
    df = lp.parse_petsc_logview(str(case / "BOUT.log.console"))
    assert "Container" not in df.index
    assert len(df) == 3


def test_a_console_with_no_report_is_a_distinct_error(tmp_path):
    """`log_view` unset must not look like a parse failure, since one is a
    choice and the other is a bug."""
    (tmp_path / "BOUT.log.console").write_text("no petsc output here\n")
    with pytest.raises(ValueError, match="No log_view report"):
        lp.parse_petsc_logview(str(tmp_path / "BOUT.log.console"))


def test_a_missing_console_says_the_run_was_launched_wrong(tmp_path):
    with pytest.raises(FileNotFoundError, match="tee"):
        lp.find_console_log(str(tmp_path))


def test_an_unparsable_row_refuses_rather_than_dropping_events(tmp_path):
    """A shifted format must fail loudly. Silently returning the rows that still
    parse would under-report the cost breakdown and look fine."""
    broken = LOGVIEW.replace("BuildTwoSided        661 1.0", "BuildTwoSided  ?? ??")
    (tmp_path / "c.txt").write_text(broken)
    with pytest.raises(ValueError, match="unparsed"):
        lp.parse_petsc_logview(str(tmp_path / "c.txt"))


def test_petsc_version_comes_from_the_report(case):
    assert lp.petsc_version(str(case / "BOUT.log.console")) == "3.23.3"


# =============================================================================
# options_left: "clean" and "never checked" are different answers
# =============================================================================
def test_no_report_is_none_not_an_empty_list(case):
    assert lp.options_left(str(case / "BOUT.log.console")) is None


def test_unused_options_are_listed(tmp_path):
    (tmp_path / "c.txt").write_text(
        "There are 2 unused database options. They are:\n"
        "Option left: name:-snes_fd_color_use_mat (no value) source: code\n"
        "Option left: name:-bogus value: 1 source: code\n"
    )
    assert lp.options_left(str(tmp_path / "c.txt")) == [
        "-snes_fd_color_use_mat", "-bogus"]


def test_trimming_keeps_the_first_of_each_object_and_the_report(tmp_path):
    """snes_view and ksp_view print once per solve. The first block says what
    PETSc built; the rest is the same text again, and on a long run it is
    gigabytes."""
    text = LOGVIEW + "".join(
        f"{kind} Object: 10 MPI processes\n  type: thing {i}\n\n"
        for i in range(5) for kind in ("SNES", "PC")
    )
    (tmp_path / "c.txt").write_text(text)
    trimmed, dropped = lp.trim_console(str(tmp_path / "c.txt"))
    assert dropped == 8  # 10 blocks, one SNES and one PC kept
    assert trimmed.count("SNES Object") == 1
    assert trimmed.count("PC Object") == 1
    assert "Total BOUT++" in trimmed  # the report itself is untouched


# =============================================================================
# BOUT.log.0
# =============================================================================
def test_wall_clock_is_computed_from_the_stamps(case):
    """Not read from the human-readable "Run time" string, so it stays a number
    and stays exact."""
    assert lp.run_header(str(case))["wall_s"] == 159.0


def test_run_header_reads_identity_and_check_level(case):
    info = lp.run_header(str(case))
    assert info["bout_version"] == "5.2.1"
    assert info["hermes_commit"].startswith("a1c3ba38")
    assert info["check_level"] == "2"
    assert info["snes_failed"] is False


def test_a_failed_run_is_recognised(tmp_path):
    """BOUT++ finalises cleanly even when the solver gave up, so the run-time
    stamps prove nothing. The marker is the evidence."""
    (tmp_path / "BOUT.log.0").write_text(BOUT_LOG + lp.SNES_FAILED_MARKER + "\n")
    assert lp.run_header(str(tmp_path))["snes_failed"] is True


def test_output_steps_reads_the_whole_table(case):
    steps = lp.output_steps(str(case))
    assert len(steps) == 2
    assert steps["rhs_evals"].sum() == 8


def test_snes_steps_reads_the_optional_failure_count(case):
    """`SNES failures` appears only on the steps that had one, so a parser that
    requires it drops every healthy step and one that ignores it loses the
    count entirely."""
    snes = lp.snes_steps(str(case))
    assert len(snes) == 2
    assert list(snes["solver_fails"]) == [0, 2]
    assert snes["nl_its"].sum() == 9
    assert snes["lin_its"].sum() == 12


def test_no_diagnose_means_no_snes_rows_not_a_crash(tmp_path):
    (tmp_path / "BOUT.log.0").write_text("BOUT++ version 5.2.1\n")
    assert lp.snes_steps(str(tmp_path)).empty


# =============================================================================
# Recipe diffing: the free consistency check on what was actually run
# =============================================================================
@pytest.fixture
def recipe_file(tmp_path):
    path = tmp_path / "A-RECIPE.txt"
    path.write_text(RECIPE_TEXT)
    return path


def test_a_bare_petsc_flag_is_a_setting(recipe_file):
    """`log_view` has no value and is meaningful by its presence, so it has to
    be recorded, or adding and removing one reads as no change."""
    settings = recipe.parse_settings(str(recipe_file))
    assert settings["petsc:log_view"] == ""
    assert settings["solver:atol"] == "1e-7"


def test_only_the_managed_sections_are_read(tmp_path):
    path = tmp_path / "r.txt"
    path.write_text("[mesh]\nfile = grid.nc\n[solver]\ntype = snes\n")
    assert recipe.parse_settings(str(path)) == {"solver:type": "snes"}


def test_an_identical_case_has_no_diffs(tmp_path, recipe_file):
    (tmp_path / "BOUT.inp").write_text(RECIPE_TEXT)
    assert recipe.diff_against_recipe(str(tmp_path), str(recipe_file)) == []


def test_the_same_number_written_differently_is_not_a_diff(tmp_path, recipe_file):
    """1e-7 and 1.0e-07 are the same tolerance. Reporting them as a deviation
    trains people to ignore the column."""
    (tmp_path / "BOUT.inp").write_text(RECIPE_TEXT.replace("1e-7", "1.0e-07"))
    assert recipe.diff_against_recipe(str(tmp_path), str(recipe_file)) == []


def test_added_and_removed_settings_are_both_reported(tmp_path, recipe_file):
    text = RECIPE_TEXT.replace("lag_jacobian = 1", "") + "scale_vars = true\n"
    (tmp_path / "BOUT.inp").write_text(text)
    diffs = recipe.diff_against_recipe(str(tmp_path), str(recipe_file))
    assert any("lag_jacobian" in d and "(absent)" in d for d in diffs)
    assert any("scale_vars" in d for d in diffs)


def test_diffs_are_ordered_so_two_runs_produce_identical_text(tmp_path, recipe_file):
    """Two runs deviating the same way must produce the same string, or the
    column cannot be grouped on."""
    (tmp_path / "BOUT.inp").write_text(
        RECIPE_TEXT.replace("atol = 1e-7", "atol = 1e-9").replace("pc_type = lu",
                                                                 "pc_type = ilu"))
    first = recipe.diff_against_recipe(str(tmp_path), str(recipe_file))
    assert first == sorted(first)


# =============================================================================
# The row lifecycle: an index a person also edits by hand
# =============================================================================
def test_a_missing_index_is_not_an_error(tmp_path):
    rows, columns = idx.read_index(str(tmp_path / "nope.tsv"))
    assert rows == []
    assert columns == idx.INDEX_COLUMNS


def test_a_declared_value_is_never_overwritten():
    """Where measurement disagrees with intent, the disagreement IS the finding
    -- a recipe drifted, or the wrong case was launched. Overwriting hides it."""
    row = {c: "" for c in idx.INDEX_COLUMNS}
    row["recipe"] = "SNES-MUMPS-1"
    filled, conflicts, unknown = idx.fill_row(
        row, {"recipe": "CVODE-2", "wall_s": 12.0}, idx.INDEX_COLUMNS)

    assert row["recipe"] == "SNES-MUMPS-1"
    assert conflicts["recipe"] == ("SNES-MUMPS-1", "CVODE-2")
    assert "wall_s" in filled


def test_a_measurement_with_no_column_is_reported_not_dropped():
    row = {c: "" for c in idx.INDEX_COLUMNS}
    _, _, unknown = idx.fill_row(row, {"invented": 1}, idx.INDEX_COLUMNS)
    assert unknown == ["invented"]


def test_floats_round_trip_exactly(tmp_path):
    """A metric re-derived from the index must equal the one that was measured;
    after the dumps are deleted there is no way to tell which was right."""
    value = 1.0 / 3.0
    row = {c: "" for c in idx.INDEX_COLUMNS}
    row["case_dir"] = "c"
    idx.fill_row(row, {"wall_s": value}, idx.INDEX_COLUMNS)
    idx.write_index(str(tmp_path / "i.tsv"), [row], idx.INDEX_COLUMNS)
    back, _ = idx.read_index(str(tmp_path / "i.tsv"))
    assert float(back[0]["wall_s"]) == value


def test_the_open_row_is_found_by_directory_name():
    rows = _rows(("case-a", idx.STATE_RECORDED), ("case-b", idx.STATE_PLANNED))
    assert idx.find_open_row(rows, "/some/where/case-b") == 1
    assert idx.find_open_row(rows, "/some/where/case-a") is None


def test_two_open_rows_for_one_directory_is_refused():
    """A directory is reused across tests, so guessing which row a finished run
    belongs to would attach results to the wrong experiment."""
    rows = _rows(("case-a", idx.STATE_PLANNED), ("case-a", idx.STATE_PLANNED))
    with pytest.raises(idx.IndexProblem, match="open"):
        idx.find_open_row(rows, "case-a")


def test_a_column_the_schema_never_heard_of_survives_a_write(tmp_path):
    """The index is hand-edited. A column someone added must not vanish because
    the writer only knows the canonical set."""
    rows, columns = idx.read_index(str(tmp_path / "nope.tsv"))
    columns = columns + ["hand_written"]
    row = {c: "" for c in columns}
    row["hand_written"] = "keep me"
    idx.write_index(str(tmp_path / "i.tsv"), [row], columns)
    back, back_columns = idx.read_index(str(tmp_path / "i.tsv"))
    assert "hand_written" in back_columns
    assert back[0]["hand_written"] == "keep me"


# =============================================================================
# Concurrency: measured across the index, never declared
# =============================================================================
def _timed(*spans):
    rows = []
    for start, wall in spans:
        row = {c: "" for c in idx.INDEX_COLUMNS}
        row["run_started"], row["wall_s"] = start, str(wall)
        rows.append(row)
    return rows


def test_a_run_alone_on_the_machine_reads_one():
    rows = _timed(("Wed Jul 29 10:00:00 2026", 60),
                  ("Wed Jul 29 12:00:00 2026", 60))
    idx.recompute_concurrency(rows)
    assert [r["concurrency"] for r in rows] == ["1.00", "1.00"]


def test_a_half_shared_run_reads_one_and_a_half():
    """Two hour-long runs offset by half an hour: each is alone for half its
    life and paired for the other half."""
    rows = _timed(("Wed Jul 29 10:00:00 2026", 3600),
                  ("Wed Jul 29 10:30:00 2026", 3600))
    idx.recompute_concurrency(rows)
    assert [r["concurrency"] for r in rows] == ["1.50", "1.50"]


def test_concurrency_is_time_weighted_not_a_count_of_overlaps():
    """A long run with two brief runs inside it, never simultaneous with each
    other. Counting distinct overlaps calls that 3 — a loading that never
    existed. Weighted by time it is 1.5, which is what the machine actually did.
    """
    rows = _timed(("Wed Jul 29 10:00:00 2026", 100),   # spans the whole window
                  ("Wed Jul 29 10:00:00 2026", 25),
                  ("Wed Jul 29 10:00:50 2026", 25))
    idx.recompute_concurrency(rows)
    assert rows[0]["concurrency"] == "1.50"


def test_concurrency_skips_a_row_it_cannot_time():
    """Declaring it would record the intended loading, which is wrong the moment
    one of three runs dies early and the other two finish alone."""
    rows = _timed(("Wed Jul 29 10:00:00 2026", 60))
    rows += _rows(("unfinished", idx.STATE_PLANNED))
    idx.recompute_concurrency(rows)
    assert rows[0]["concurrency"] == "1.00"
    assert rows[1]["concurrency"] == ""


# =============================================================================
# Resolving a row back to the directory it ran in
# =============================================================================
def test_a_row_with_no_case_dir_resolves_to_nothing(tmp_path):
    """os.path.join(root, "") is the root, which exists. A legacy row carries no
    case_dir, so without an explicit check it "verifies" against a directory
    that is not a case and reports a match, because an empty record has nothing
    to disagree with. This masked nine rows in the first run of the check."""

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "cli"))
    from verify_store import find_case

    assert find_case("", [str(tmp_path)]) is None
    assert find_case("   ", [str(tmp_path)]) is None

    (tmp_path / "a-real-case").mkdir()
    assert find_case("a-real-case", [str(tmp_path)]) == str(tmp_path / "a-real-case")
