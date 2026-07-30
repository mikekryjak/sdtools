"""Parsing of the text a Hermes-3 run leaves behind.

Two files, with different jobs. `BOUT.log.0` holds the per-output-step table and
the SNES per-step lines. The captured console (`BOUT.log.console`) holds
everything PETSc prints at finalize -- the `log_view` profiling report,
`options_left`, `snes_view`/`ksp_view` and the PETSc version -- none of which
exists anywhere else once the run is over.

Text parsing is the sanctioned route for solver output: neither xhermes nor the
rest of sdtools covers it. Dump data is a different matter and must go through
xhermes.
"""

import datetime
import os
import re

import pandas as pd

# PETSc writes log_view, options_left, snes_view and its version banner to
# stdout at finalize. sdrun.py execs into mpirun, so unless the launch line
# tees stdout to this file beside the case, all of it is discarded.
CONSOLE_LOG = "BOUT.log.console"

# Columns PETSc prints after the event name, in order. The event name can
# contain spaces ("Total BOUT++"), so rows are split from the right by this
# count rather than by fixed character widths.
EVENT_COLUMNS = [
    "count_max",
    "count_ratio",
    "time_max",  # s
    "time_ratio",
    "flop_max",
    "flop_ratio",
    "mess",
    "avglen",
    "reduct",
    "global_t",  # %
    "global_f",
    "global_m",
    "global_l",
    "global_r",
    "stage_t",  # %
    "stage_f",
    "stage_m",
    "stage_l",
    "stage_r",
    "total",  # Mflop/s
]


def find_console_log(case_dir):
    """
    Path to the captured console output for a case. Raises if it is missing,
    since that means the run was launched without teeing stdout and its PETSc
    output no longer exists anywhere.
    """

    path = os.path.join(case_dir, CONSOLE_LOG)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No {CONSOLE_LOG} in {case_dir}. PETSc writes its log_view report to"
            " stdout at finalize and nothing else keeps it, so the run must be"
            f" launched as '... 2>&1 | tee <case>/{CONSOLE_LOG}'."
        )
    return path


def _find_event_header(lines):
    """Index of the log_view event table header, or None if there is no report."""

    for i, line in enumerate(lines):
        if line.startswith("Event") and "Count" in line and "Time (sec)" in line:
            return i
    return None


def _parse_event_rows(lines):
    """
    Rows of the event timing table, as {event name: [values]}. Reads the first
    event stage only — BOUT++ does not push additional stages. Returns the rows
    and any lines inside the table that could not be parsed, so that a format
    change shows up as skipped lines instead of silently missing events.
    """

    rows = {}
    skipped = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        if not in_table:
            # Rows begin after the stage marker, not after the header: the
            # header is followed by a rule, a blank line and the marker.
            if stripped.startswith("--- Event Stage"):
                in_table = True
            continue

        if stripped and set(stripped) == {"-"}:
            break  # closing rule of the timing table
        if not stripped:
            continue

        tokens = stripped.split()
        if len(tokens) <= len(EVENT_COLUMNS):
            skipped.append(stripped)
            continue

        name = " ".join(tokens[: -len(EVENT_COLUMNS)])
        try:
            rows[name] = [float(t) for t in tokens[-len(EVENT_COLUMNS) :]]
        except ValueError:
            skipped.append(stripped)

    return rows, skipped


def parse_petsc_logview(path):
    """
    Extracts the event timing table (the CPU cost breakdown) from PETSc's
    log_view report. Takes either a case directory or a path to the file
    holding the console output. Returns a dataframe indexed by event name.
    """

    if os.path.isdir(path):
        path = find_console_log(path)

    with open(path, "r") as f:
        lines = f.readlines()

    header = _find_event_header(lines)
    if header is None:
        raise ValueError(
            f"No log_view report in {path}. The console was captured but PETSc"
            " printed no profiling report — check that log_view is set in the"
            " [petsc] section of BOUT.inp."
        )

    rows, skipped = _parse_event_rows(lines[header:])
    if not rows:
        raise ValueError(f"Found a log_view report in {path} but no event rows in it.")
    if skipped:
        raise ValueError(
            f"{len(skipped)} unparsed line(s) inside the log_view event table in"
            f" {path}, first: {skipped[0]!r}. The report format has changed;"
            " parsing it as-is would drop events silently."
        )

    return pd.DataFrame.from_dict(rows, orient="index", columns=EVENT_COLUMNS)


# =============================================================================
# The rest of the captured console
# =============================================================================
def _read_text(path, filename=None):
    """Text of a file, resolving a case directory to one of its logs."""

    if os.path.isdir(path):
        path = os.path.join(path, filename) if filename else find_console_log(path)
    with open(path, "r", errors="replace") as f:
        return f.read()


def petsc_version(path):
    """PETSc version from the log_view banner, or None. BOUT.log.0 records only
    that PETSc support was compiled in, never which version ran."""

    m = re.search(r"Using PETSc (?:Release )?Version\s+([0-9.]+)", _read_text(path))
    return m.group(1) if m else None


def options_left(path):
    """
    Options PETSc parsed but never used, as a list of names. `[petsc]` has no
    whitelist, so a misspelled option is silently ignored and the recipe quietly
    does not do what it says -- this is the only thing that catches it.

    Returns an empty list if the report is present and clean, and None if
    options_left was not enabled, since "clean" and "never checked" must not
    look the same.
    """

    text = _read_text(path)
    if "unused database option" not in text:
        return None
    return re.findall(r"^Option left: name:(\S+)", text, re.MULTILINE)


def first_view_block(path, marker="SNES Object"):
    """
    The first `snes_view` / `ksp_view` block. These print once per solve, not
    once per run, so only the first is kept -- on a long run the rest is
    gigabytes of identical text.
    """

    text = _read_text(path)
    start = text.find(marker)
    if start == -1:
        return None

    # The block ends at the first line that is neither blank nor indented,
    # since PETSc indents every line of an object dump below its header.
    lines = text[start:].splitlines()
    block = [lines[0]]
    for line in lines[1:]:
        if line and not line[0].isspace():
            break
        block.append(line)
    return "\n".join(block).rstrip()


_OBJECT_HEADER = re.compile(r"^([A-Za-z_]+) Object")


def trim_console(path):
    """
    The console with repeated PETSc object dumps removed, and how many went.

    `snes_view` and `ksp_view` print after every solve, not once per run: a
    2.5 minute test produces 66 and 390 of them, and a 34 hour run would leave a
    console of order a gigabyte. The first of each says what PETSc built, which
    is the whole point of asking; the rest are identical. Everything else --
    log_view, options_left, the version banner, the run's own output -- is kept
    untouched, so the file stays usable as evidence.
    """

    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    kept, dropped, seen = [], 0, set()
    i = 0
    while i < len(lines):
        # Any PETSc object dump, not a fixed list: a ksp_view emits PC and Mat
        # objects as well as the KSP itself, and naming only the obvious two
        # leaves the bulk behind. Nested objects are indented, so an
        # unindented header is a top-level block.
        header = _OBJECT_HEADER.match(lines[i])
        if header is None:
            kept.append(lines[i])
            i += 1
            continue
        marker = header.group(1)

        block_end = i + 1
        while block_end < len(lines) and (
            not lines[block_end].strip() or lines[block_end][0].isspace()
        ):
            block_end += 1

        if marker in seen:
            dropped += 1
        else:
            seen.add(marker)
            kept.extend(lines[i:block_end])
        i = block_end

    if dropped:
        kept.append(
            f"\n[{dropped} repeated PETSc object dump(s) removed at extraction:"
            " snes_view and ksp_view print once per solve, and only the first"
            " block of each carries information]\n"
        )

    return "".join(kept), dropped


def run_banner(path):
    """
    Slot and core range from the sdrun launch banner in the console. The slot is
    measured here rather than declared, since a run can be launched into a
    different slot than the one that was planned.
    """

    text = _read_text(path)
    out = {}

    m = re.search(
        r"Cores\s+(?P<cores>[\d-]+)\s+\(slot (?P<slot>\d+), (?P<procs>\d+) procs\)",
        text,
    )
    if m:
        out["core_range"] = m.group("cores")
        out["slot"] = int(m.group("slot"))
        out["procs"] = int(m.group("procs"))

    # The banner records which branch the executable was built from. That is
    # better than asking git which branches contain the commit, since a commit
    # can sit on several and only one of them was built.
    m = re.search(r"^\s*Build\s+(?P<exe>\S+)\s+@\s+(?P<branch>[^,\s]+)", text, re.MULTILINE)
    if m:
        out["build_exe"] = m.group("exe")
        out["hermes_branch"] = m.group("branch")

    return out


# =============================================================================
# BOUT.log.0
# =============================================================================
LOG_TIME_FMT = "%a %b %d %H:%M:%S %Y"

# BOUT++ exits cleanly even when the solver gave up, so the run-time stamps are
# not evidence of success. This is.
SNES_FAILED_MARKER = "======== SNES failed ========="

_STEP_COLUMNS = [
    "sim_time",
    "rhs_evals",
    "wall_time",
    "calc_pct",
    "inv_pct",
    "comm_pct",
    "io_pct",
    "solver_pct",
]

_SNES_STEP = re.compile(
    r"Time:\s*(?P<time>[-\d.eE+]+),\s*"
    r"timestep:\s*(?P<timestep>[-\d.eE+]+),\s*"
    r"nl iter:\s*(?P<nl_its>\d+),\s*"
    r"lin iter:\s*(?P<lin_its>\d+),\s*"
    r"reason:\s*(?P<reason>-?\d+)"
    r"(?:,\s*SNES failures:\s*(?P<solver_fails>\d+))?"
)


def run_header(case_dir):
    """
    Identity and wall clock from BOUT.log.0. `wall_s` is computed from the start
    and finish stamps rather than read from the human-readable "Run time"
    string, so it stays exact and stays a number.
    """

    text = _read_text(case_dir, "BOUT.log.0")

    def search(pattern):
        m = re.search(pattern, text, re.MULTILINE)
        return m.group(1).strip() if m else None

    info = {
        "bout_version": search(r"^BOUT\+\+ version (.+)$"),
        "bout_commit": search(r"^Revision:\s*([0-9a-fA-F]+)"),
        "hermes_commit": search(r"^Git Version of Hermes:\s*([0-9a-fA-F]+)"),
        "run_started": search(r"^Run started at\s*:\s*(.+)$"),
        "run_finished": search(r"^Run finished at\s*:\s*(.+)$"),
        "check_level": search(r"Runtime error checking enabled, level (\d+)"),
        "snes_failed": SNES_FAILED_MARKER in text,
    }
    if info["check_level"] is None and "Runtime error checking disabled" in text:
        info["check_level"] = "0"

    info["wall_s"] = None
    info["started_at"] = None
    if info["run_started"]:
        try:
            info["started_at"] = datetime.datetime.strptime(
                info["run_started"], LOG_TIME_FMT
            )
        except ValueError:
            pass
    if info["started_at"] and info["run_finished"]:
        try:
            end = datetime.datetime.strptime(info["run_finished"], LOG_TIME_FMT)
            info["wall_s"] = (end - info["started_at"]).total_seconds()
        except ValueError:
            pass

    return info


def output_steps(case_dir):
    """
    The per-output-step table BOUT++ prints, as a dataframe. Present for every
    solver, so it is the one cost series that compares across all of them.
    """

    text = _read_text(case_dir, "BOUT.log.0")
    header = text.find("Sim Time")
    if header == -1:
        return pd.DataFrame(columns=_STEP_COLUMNS)

    rows = []
    for line in text[header:].splitlines()[1:]:
        tokens = line.split()
        if len(tokens) != len(_STEP_COLUMNS):
            continue
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            continue

    return pd.DataFrame(rows, columns=_STEP_COLUMNS)


def snes_steps(case_dir):
    """
    The SNES per-step lines, as a dataframe. Requires `[solver] diagnose = true`
    and returns an empty frame otherwise.

    One row per INTERNAL timestep, not per output step: once the timestep grows
    past the output interval a single step spans several outputs, so there are
    usually fewer rows here than output steps and the two series must not be
    aligned by index. `solver_fails` is the count SNES reports, carried forward
    on the steps where it prints nothing.
    """

    text = _read_text(case_dir, "BOUT.log.0")
    rows = []
    for m in _SNES_STEP.finditer(text):
        row = m.groupdict()
        rows.append(
            {
                "time": float(row["time"]),
                "timestep": float(row["timestep"]),
                "nl_its": int(row["nl_its"]),
                "lin_its": int(row["lin_its"]),
                "reason": int(row["reason"]),
                "solver_fails": int(row["solver_fails"] or 0),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["time", "timestep", "nl_its", "lin_its", "reason", "solver_fails"],
    )
