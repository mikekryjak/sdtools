"""Run provenance: what a case was actually run with, and how cases differ.

Everything here reads a case DIRECTORY (BOUT.log.0, BOUT.settings, BOUT.inp) --
no dataset needed, so a study cover can be built even for a case that failed to
load.
"""

import os
import re
import datetime
import subprocess
from itertools import islice

from boutdata.data import BoutOptionsFile


# =============================================================================
# Analysis-time provenance (which code built the PDF)
# =============================================================================
def git_describe(repo):
    """"<sha> (<branch>)[+dirty]" for a repo, or "?" if it isn't one."""
    def run(args):
        try:
            return subprocess.check_output(
                ["git", "-C", repo, *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return ""
    sha = run(["rev-parse", "--short", "HEAD"]) or "?"
    branch = run(["rev-parse", "--abbrev-ref", "HEAD"]) or "?"
    dirty = "+dirty" if run(["status", "--porcelain"]) else ""
    return f"{sha} ({branch}){dirty}"


# =============================================================================
# Run-time provenance (which code ran the case)
# =============================================================================
def _log_head(case_dir, nlines=400):
    log = os.path.join(case_dir, "BOUT.log.0") if case_dir else None
    if not (log and os.path.exists(log)):
        return None, None
    with open(log, errors="ignore") as f:
        return log, "".join(islice(f, nlines))


def run_info(case_dir):
    """Per-case run provenance parsed from BOUT.log.0.

    Returns dict(hermes=, bout=, date=), each "?" if unavailable.
    """
    info = dict(hermes="?", bout="?", date="?")
    log, head = _log_head(case_dir)
    if head is None:
        return info
    info["date"] = datetime.datetime.fromtimestamp(
        os.path.getmtime(log)
    ).strftime("%Y-%m-%d")
    m = re.search(r"Git Version of Hermes:\s*([0-9a-f]+)", head)
    if m:
        info["hermes"] = m.group(1)[:8]
    m = re.search(r"BOUT\+\+ version\s+(\S+)", head)
    if m:
        info["bout"] = m.group(1)
    return info


def check_level(case_dir):
    """BOUT++ runtime error-checking (CHECK) level from BOUT.log.0, else '?'.

    Wall times are only comparable across cases at the SAME CHECK level (and
    core count) -- plot_runtime_breakdown annotates each bar with this and
    flags a set that mixes levels.
    """
    _, head = _log_head(case_dir)
    if head is None:
        return "?"
    m = re.search(r"Runtime error checking enabled, level (\d+)", head)
    if m:
        return m.group(1)
    if re.search(r"Runtime error checking disabled", head):
        return "0"
    return "?"


def run_finished(case_dir):
    """True iff the case has a FINALISED BOUT.settings (BoutFinalise ran).

    BOUT writes BOUT.settings twice: an init-time stub carrying only the
    values from BOUT.inp (no code defaults, [run] has `started` but not
    `finished`), then at the end BoutFinalise overwrites it with the fully
    resolved version -- every code default filled in, plus a [run] `finished`
    stamp. Only that final file is authoritative; an interrupted or
    still-running case leaves the stub. Presence of the `finished =` key is the
    reliable discriminator (written by setRunFinishInfo just before the
    finalise overwrite).
    """
    p = os.path.join(case_dir, "BOUT.settings") if case_dir else None
    if not (p and os.path.exists(p)):
        return False
    with open(p, errors="ignore") as f:
        return re.search(r"(?m)^\s*finished\s*=", f.read()) is not None


def run_status(case_dir):
    """Coarse run state for the cover's CASES table.

    finished   -- finalised BOUT.settings present (authoritative)
    unfinished -- BOUT.settings is the init stub (interrupted or still running)
    no-run     -- no BOUT.settings at all (prepared but never launched)
    """
    p = os.path.join(case_dir, "BOUT.settings") if case_dir else None
    if not (p and os.path.exists(p)):
        return "no-run"
    return "finished" if run_finished(case_dir) else "unfinished"


# =============================================================================
# Option diffing
# =============================================================================
# Provenance/bookkeeping keys that differ trivially on every case and would
# bury the real solver/physics differences. DENYLIST, not allowlist: anything
# not matched here is compared, so a new knob can never be silently hidden.
_PROVENANCE_EXACT = {"datadir", "run_id", "run_restart_from", "optionfile",
                     "settingsfile", "append"}
_PROVENANCE_SECTIONS = ("run:",)  # run:started, run:finished, run:run_id, ...


def _is_provenance(key):
    if key in _PROVENANCE_EXACT:
        return True
    if key.split(":")[-1] == "path":  # output:path, restart_files:path, ...
        return True
    return any(key.startswith(p) for p in _PROVENANCE_SECTIONS)


def _flatten_options(d, prefix=""):
    """Flatten the nested option tree to {"section:...:key": "value"} leaves."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}:{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_options(v, key))
        else:
            out[key] = str(v).strip()
    return out


def case_options(case_dir):
    """Flattened BOUT.inp for a case -- the INPUT, not what the run resolved.

    Only a fallback for options_used() when a case has no log (never launched).
    Complete and interruption-proof, but carries no code defaults.
    """
    if not case_dir:
        return {}
    p = os.path.join(case_dir, "BOUT.inp")
    if os.path.exists(p):
        try:
            return _flatten_options(
                BoutOptionsFile(p, recalculate_xyz=False).as_dict()
            )
        except Exception:
            pass
    return {}


# BOUT logs every option AS IT IS READ:  "\tOption mesh:file = x.nc (source)"
# where source is `default`, `user_default`, a BOUT.inp path, `Command line`,
# the grid file, or empty.
_OPTION_LINE = re.compile(r"^\s*Option\s+(\S+)\s*=\s*(.*?)\s*\(([^()]*)\)\s*$")

# The option dump sits in the first few hundred lines even of a 100 MB log
# (measured: deepest last-option line 19_900 across a 120-log sample), but the
# log can be millions of lines, so stop once the options are clearly done
# rather than reading the whole file.
_OPTION_SCAN_MAX = 60_000
_OPTION_SCAN_GAP = 5_000


def _parse_log_options(case_dir):
    """{key: value} for every option BOUT logged READING, last occurrence wins.

    A key is logged more than once when BOUT re-reads it after defaults are
    applied (MXG appears as 0 then 2); last-wins is what matches the finalised
    BOUT.settings (12/13 against 9/13 for first-wins on a calibration case).
    """
    log = os.path.join(case_dir, "BOUT.log.0") if case_dir else None
    if not (log and os.path.exists(log)):
        return {}
    values, last_hit = {}, 0
    with open(log, errors="ignore") as f:
        for i, line in enumerate(f):
            if i > _OPTION_SCAN_MAX or (values and i - last_hit > _OPTION_SCAN_GAP):
                break
            m = _OPTION_LINE.match(line)
            if m:
                values[m.group(1)] = m.group(2)
                last_hit = i
    return values


def options_used(case_dir, with_sources=False):
    """The best available record of a case's options, layered by trustworthiness.

    No single file is both complete and interruption-proof, so three are
    combined. Later layers OVERRIDE earlier ones; the last layer only FILLS
    GAPS, so nothing it contains can ever corrupt a value:

      1. BOUT.inp -- what was ASKED FOR. Always present and interruption-proof,
         but carries no code defaults (~140 keys against a resolved ~360). It
         is the bottom layer so that an option SET but never READ by this build
         still shows up: silently-ignored options are a known trap, and they
         are invisible in the log precisely because nothing read them.
      2. BOUT.log.0 -- what was actually READ, logged at read time, so an
         interrupted run still has essentially the full set (measured: 271-322
         keys). This is the authoritative layer and overrides BOUT.inp.
      3. BOUT.settings, GAP-FILL ONLY and only when FINALISED -- recovers the
         handful of values the code FORCES into the options tree rather than
         reading, which therefore appear in no log and no input:
         hermes:slope_limiter, hermes:conduction_method, hermes:revision.
         Those are the very knobs some campaigns are built to compare, so
         dropping them would be worse than the risk of including them -- and
         the risk is nil here because a non-finalised stub is never read and
         this layer can never override layers 1-2.

    Why BOUT.settings is never a BASIS: it is written twice -- an init-time
    stub, then overwritten with the fully resolved set by BoutFinalise (see
    bout++.cxx:213 and BoutFinalise's "This overwrites the file written during
    initialisation"). An interrupted run leaves a STUB, not a missing file,
    carrying almost no defaults (measured: 95-143 keys against a finished run's
    317-384), so comparing a stub against a finished settings manufactures a
    phantom difference for every defaulted option.

    Returns {key: value}, or ({key: value}, {key: layer}) with
    with_sources=True, where layer is "BOUT.inp" / "BOUT.log.0" /
    "BOUT.settings".
    """
    values, sources = {}, {}

    for k, v in case_options(case_dir).items():
        values[k], sources[k] = v, "BOUT.inp"

    for k, v in _parse_log_options(case_dir).items():
        values[k], sources[k] = v, "BOUT.log.0"

    if case_dir and run_finished(case_dir):
        p = os.path.join(case_dir, "BOUT.settings")
        try:
            resolved = _flatten_options(
                BoutOptionsFile(p, recalculate_xyz=False).as_dict()
            )
        except Exception:
            resolved = {}
        for k, v in resolved.items():
            if k not in values:  # gap-fill only -- never override
                values[k], sources[k] = v, "BOUT.settings"

    return (values, sources) if with_sources else values


def _priority_rank(key, priority):
    """0 if `key` matches any entry in `priority`, else 1.

    An entry matches either as a leading prefix of the full flattened key
    ("solver:", "petsc:") or as an exact bare option name matched against the
    key's last segment ("flux_limit" matches "d:flux_limit"). One knob then
    serves both styles of priority list.
    """
    leaf = key.split(":")[-1]
    for p in priority:
        if key.startswith(p) or leaf == p:
            return 0
    return 1


ABSENT = "<absent>"
VARIES = "varies"


def _comparable(value):
    """Value normalised for COMPARISON only (never for display).

    BOUT logs booleans either way round depending on where they were read
    (`0` / `false`), so compare them as one. Nothing else is touched, and a
    genuine 0-vs-1 difference still reads as false-vs-true, i.e. still differs.
    """
    v = value.strip()
    return {"0": "false", "1": "true"}.get(v, v)


def collapse_options(option_sets):
    """Reduce several cases' options to ONE column: the shared value, or VARIES.

    Used where a column stands for a GROUP of cases rather than one case (a
    scan series). A key missing from some members reads as ABSENT for those,
    so a key present in only part of the group collapses to VARIES.
    """
    option_sets = list(option_sets)
    if not option_sets:
        return {}
    allkeys = set().union(*[set(o) for o in option_sets])
    out = {}
    for k in allkeys:
        seen = {_comparable(o.get(k, ABSENT)) for o in option_sets}
        out[k] = option_sets[0].get(k, ABSENT) if len(seen) == 1 else VARIES
    return out


def diff_option_sets(per_label, priority=(), sources=None):
    """THE option-diff core. Every campaign type goes through this.

    per_label : ordered {column label: {option key: value}}. A column is one
        case (case campaigns) or one collapsed series (scan campaigns) -- see
        collapse_options. Where the values come from is the caller's business;
        options_used() is the right source for both.
    priority : DISPLAY-ORDER HINT ONLY -- never a filter.
    sources : optional {column label: {key: layer}} from options_used(
        with_sources=True). Used only to suppress one specific kind of noise --
        see `unrecorded` below.

    Returns (differing_keys, filled_per_label, unrecorded).

    `unrecorded` holds keys that LOOK like differences but are really just
    missing data: values the code FORCES into the options tree (build flags
    such as hermes:slope_limiter) are recoverable only from a finalised
    BOUT.settings, so a study mixing finished and still-running cases would
    otherwise show a row of "<absent>" for each of them -- around 20 rows of
    pure noise, drowning the real differences. A key is moved here when it is
    absent from at least one column AND every column that has it got it from
    the BOUT.settings layer. Callers should report the list in one line rather
    than tabulate it. With sources=None nothing is suppressed.

    A key absent from a column reads as ABSENT, which itself counts as a
    difference.

    History (do not "optimise" this back): the priority list was once an
    ALLOWLIST -- the only keys compared -- which silently hid any knob not
    named in it. A campaign that renamed its limiter options then showed "no
    difference" on every cover. Compare everything; rank for display only.
    """
    labels = list(per_label)
    per = {l: dict(per_label[l]) for l in labels}
    allkeys = set().union(*[set(v) for v in per.values()]) if per else set()
    allkeys = {k for k in allkeys if not _is_provenance(k)}
    for l in labels:  # fill gaps so the render can index every (column, key)
        for k in allkeys:
            per[l].setdefault(k, ABSENT)

    differing = [k for k in allkeys
                 if len({_comparable(per[l][k]) for l in labels}) > 1]

    unrecorded = []
    if sources:
        keep = []
        for k in differing:
            missing = [l for l in labels if per[l][k] == ABSENT]
            if missing and all(
                sources.get(l, {}).get(k) == "BOUT.settings"
                for l in labels if per[l][k] != ABSENT
            ):
                unrecorded.append(k)
            else:
                keep.append(k)
        differing = keep

    differing.sort(key=lambda k: (_priority_rank(k, priority), k))
    unrecorded.sort(key=lambda k: (_priority_rank(k, priority), k))
    return differing, per, unrecorded


def param_diff(case_dirs, priority=()):
    """Diff the options ACTUALLY USED across cases (see options_used).

    case_dirs : ordered {case name: case directory}.
    Returns (differing_keys, {name: {key: value}}, unrecorded, {name: has_log}).
    """
    per, srcs, has_log = {}, {}, {}
    for name, d in case_dirs.items():
        per[name], srcs[name] = options_used(d, with_sources=True)
        log = os.path.join(d, "BOUT.log.0") if d else None
        has_log[name] = bool(log and os.path.exists(log))
    differing, filled, unrecorded = diff_option_sets(
        per, priority=priority, sources=srcs
    )
    return differing, filled, unrecorded, has_log
