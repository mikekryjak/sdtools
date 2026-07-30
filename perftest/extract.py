"""Turn a finished case directory into a durable record.

One entry point, `extract_case`. It reads a case directory, writes a bundle of
evidence and parsed tables, fills in the one index row that was opened for it,
and reports whether everything it needed was actually there -- because the
answer to that question is what decides whether the dumps can be deleted.

Two rules shape the whole module.

Every number analysis could want is stored as a number. Text goes into the
bundle as evidence, but nothing downstream should ever have to parse it again:
once the dumps are gone the run cannot be re-read, so a figure left in prose is
a figure lost.

Nothing is invented. A value that could not be measured is left empty, never
defaulted and never guessed, because an empty cell is a question somebody can
answer later and a wrong cell is a conclusion nobody will re-check.
"""

import datetime
import getpass
import os
import re
import shutil
import socket

from . import index as idx
from . import recipe

# Full round-trip precision. A metric re-derived from a truncated table would
# silently disagree with the same metric read from the dump, and after the dumps
# are deleted there is no way to tell which was right.
FLOAT_FORMAT = "%.17g"

BUNDLE_FILES = ["BOUT.inp", "BOUT.settings", "BOUT.log.0", "BOUT.log.console"]

# log_view event names, and the schema column each one's time share feeds.
TIME_SHARES = {
    "t_jac_frac": "SNESJacobianEval",
    "t_pcsetup_frac": "PCSetUp",
    "t_ksp_frac": "KSPSolve",
    "t_func_frac": "SNESFunctionEval",
}


class Report:
    """What the extraction found, and whether the case is safe to delete."""

    def __init__(self, case_dir):
        self.case_dir = case_dir
        self.test_id = None
        self.bundle = None
        self.record = {}
        self.problems = []  # block deletion
        self.warnings = []  # worth knowing, do not block
        self.filled = []
        self.conflicts = {}

    @property
    def ok(self):
        return not self.problems

    def __str__(self):
        lines = [f"{self.case_dir}"]
        lines.append(f"  test_id  {self.test_id or '(none)'}")
        lines.append(f"  bundle   {self.bundle or '(not written)'}")
        if self.filled:
            lines.append(f"  filled   {len(self.filled)} cells")
        for key, (declared, measured) in self.conflicts.items():
            lines.append(f"  CONFLICT {key}: declared {declared!r}, measured {measured!r}")
        for w in self.warnings:
            lines.append(f"  warning  {w}")
        for p in self.problems:
            lines.append(f"  PROBLEM  {p}")
        verdict = (
            "extraction validated - case directory is safe to delete"
            if self.ok
            else "extraction incomplete - DO NOT delete the case directory"
        )
        lines.append(f"  => {verdict}")
        return "\n".join(lines)


# =============================================================================
# Small readers
# =============================================================================
def _settings_finished(case_dir):
    """
    True if BoutFinalise ran. BOUT++ writes BOUT.settings twice: a stub at
    startup, then a fully resolved version carrying a `finished` stamp at the
    end. Only the second means the run reached its own exit.
    """

    path = os.path.join(case_dir, "BOUT.settings")
    if not os.path.exists(path):
        return False
    with open(path, errors="ignore") as f:
        return re.search(r"(?m)^\s*finished\s*=", f.read()) is not None


def _inp_value(case_dir, section, key):
    """One value from BOUT.inp, or None. Enough for the few settings needed to
    find the grid file and the expected length; not a general options reader."""

    path = os.path.join(case_dir, "BOUT.inp")
    if not os.path.exists(path):
        return None

    current = ""
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip().lower()
                continue
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip().lower() == key.lower() and current == section.lower():
                return value.strip()
    return None


def _resolve_grid(case_dir, grid_path=None):
    """
    Path to the grid file. Hermes resolves the grid relative to the directory a
    run was LAUNCHED from, not the case directory, so the default lives one
    level up.
    """

    if grid_path:
        return grid_path if os.path.exists(grid_path) else None

    name = _inp_value(case_dir, "mesh", "file")
    if not name:
        return None
    name = name.strip("\"'")

    for candidate in (
        os.path.join(case_dir, name),
        os.path.join(os.path.dirname(os.path.abspath(case_dir)), name),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _infer_test(case_dir):
    """Test name from the directory, which is named `<test>-<date>[-<desc>]`."""

    return idx.case_key(case_dir).split("-")[0]


# =============================================================================
# The dump
# =============================================================================
def _read_dump(case_dir, grid_path, report):
    """
    Everything the record needs from the dumps, via xhermes.

    Never xbout or xarray directly: that path is where the guard-cell and
    normalisation bugs come from. xhermes also unnormalises to SI, which is what
    makes the physics numbers comparable to anything.
    """

    import xhermes

    ds = xhermes.open(case_dir, geometry="toroidal", gridfilepath=grid_path)
    meta = dict(ds.metadata) if hasattr(ds, "metadata") else {}
    meta.update({k: v for k, v in ds.attrs.items() if k not in meta})

    out = {
        "run_id": meta.get("run_id"),
        "seed": meta.get("run_restart_from"),
        "limiter": meta.get("HERMES_SLOPE_LIMITER"),
        "check_level": meta.get("use_check_level"),
    }
    # bout_version deliberately not taken from the dump: BOUT_VERSION is stored
    # as a float there (5.21), while BOUT.log.0 carries the exact string (5.2.1).

    nxpe, nype = meta.get("NXPE"), meta.get("NYPE")
    if nxpe and nype:
        out["cores"] = int(nxpe) * int(nype)
        out["decomposition"] = f"{int(nxpe)}x{int(nype)}"

    # Grid fingerprint, so a renamed grid file is still recognisable.
    shape = [meta.get(k) for k in ("nx", "ny", "nz", "ixseps1", "ixseps2")]
    if all(v is not None for v in shape):
        out["grid_shape"] = "nx{0}_ny{1}_nz{2}_ix{3}-{4}".format(*[int(v) for v in shape])

    # RHS evaluations. Per output step, NOT cumulative: the final value on a
    # finished run is 1, so reading it instead of summing records a
    # one-evaluation run and nothing looks wrong.
    if "ncalls" in ds:
        out["ncalls"] = int(ds["ncalls"].values.sum())

    # Simulated time this test covered. The span, not the endpoint: a restarted
    # run begins at its seed's clock, and the span is both what the speed metric
    # divides and what says how far a failed run got.
    # A run with a single output step covered no simulated time at all, which is
    # a real measurement of a run that got nowhere -- recorded as 0, not left
    # empty, since empty would read as "not known".
    if "t" in ds:
        import numpy as np

        # atleast_1d because a run that wrote a single output step has its time
        # dimension squeezed away entirely, and that is exactly the failed run
        # whose reach we most want on the record.
        t = np.atleast_1d(ds["t"].values)
        out["sim_time_ms"] = float(t[-1] - t[0]) * 1e3
        out["n_output_steps"] = int(t.size)

    if _interior_mask(ds, report) is None:
        report.problems.append("field reductions skipped: no interior mask")
    out.update(_residual_metrics(ds, out.get("ncalls")))
    out.update(_physics_gates(ds, report))
    out["_series"] = _dump_series(ds)
    out["_resid_regions"] = _residual_regions(ds, report)
    out["_ddt"] = _ddt_series(ds)
    out["_physics"] = _physics_series(ds, report)

    ds.close()
    return out


# Per-output-step quantities worth keeping. Scalars in time, so the whole
# history costs a few kilobytes -- and it is the only record of how a run
# behaved once the dumps are gone.
SERIES_FIELDS = [
    "ncalls",
    "ncalls_e",
    "ncalls_i",
    "wtime",
    "wtime_rhs",
    "wtime_invert",
    "wtime_comms",
    "wtime_io",
    "wall_time",
    "snes_global_residual",
    "cvode_nsteps",
    "cvode_nfevals",
    "cvode_nniters",
    "cvode_nliters",
    "cvode_num_fails",
    "cvode_nonlin_fails",
]


# Diagnostic zones for a connected double null, label -> (radial, poloidal)
# xhermes selector pair. Verified to tile the interior exactly on the perftest
# grids: the eleven regions sum to nx-2*MXG by ny cells, and the extractor
# records the uncovered remainder anyway so a grid they do not tile says so
# rather than quietly dropping residual.
# Below this share of the domain residual norm a (region, equation) is not
# stored -- see _residual_regions.
RESID_REGION_FLOOR = 1e-6

RESID_REGIONS = {
    "core": ("core", "core"),
    "SOL inner lower upstream": ("sol", "inner_lower_upstream"),
    "SOL outer lower upstream": ("sol", "outer_lower_upstream"),
    "SOL inner upper upstream": ("sol", "inner_upper_upstream"),
    "SOL outer upper upstream": ("sol", "outer_upper_upstream"),
    "SOL inner lower leg": ("sol", "inner_lower_divertor"),
    "SOL outer lower leg": ("sol", "outer_lower_divertor"),
    "SOL inner upper leg": ("sol", "inner_upper_divertor"),
    "SOL outer upper leg": ("sol", "outer_upper_divertor"),
    "lower PFR": ("pfr", "lower_pfr"),
    "upper PFR": ("pfr", "upper_pfr"),
}


def _interior_mask(ds, report=None):
    """True on the cells the solver actually evolves, False on guard cells.

    Built from xhermes' own `clear_guards`, not from index arithmetic, because
    a connected double null carries poloidal guards in the MIDDLE of theta as
    well as at its ends -- at the upper targets -- and trimming MYG off each
    end silently leaves those in. It is also the definition the shared ddt
    plotting uses, so a report and a live monitor agree by construction.

    Taken from the geometry rather than from a physics field: J is finite in
    every cell including the guards, so anything NaN after clearing is a guard
    cell and nothing else.

    None means the mask could not be built. Callers must treat that as a
    failure, never as "there were no guards to remove".
    """

    import numpy as np

    for name in ("J", "dx", "dy"):
        if name not in ds:
            continue
        try:
            cleared = ds[name].hermes.clear_guards()
        except Exception as exc:  # noqa: BLE001
            if report is not None:
                report.warnings.append(f"clear_guards failed on {name}: {exc}")
            continue
        mask = np.isfinite(cleared)
        if "t" in mask.dims:
            mask = mask.isel(t=0, drop=True)
        return mask

    if report is not None:
        report.problems.append(
            "guard cells cannot be identified: no geometry field to build an"
            " interior mask from, so no field reduction can be trusted"
        )
    return None


def _interior(field, mask):
    """`field` with its guard cells set aside.

    NOT optional tidying. The solver's state vector holds interior cells only,
    so a guard cell has no residual and no rate of change -- but the saved
    fields carry whatever was in that memory, and on these runs that is up to
    1e15 against an interior value of 1e-3. Reducing over the raw array ranks
    the equations by which one has the worst garbage in its guard cells (F20).

    Guards become NaN rather than being trimmed, so every reduction over the
    result must stay NaN-skipping -- which xarray's sum, mean and max are by
    default. Trimming would not survive a double null, whose guards are not all
    at the edges.
    """

    return field if mask is None else field.where(mask)


def _resid_fields(ds):
    return sorted(v for v in ds.data_vars if str(v).startswith("resid_"))


def _residual_shares(ds):
    """
    Per-equation share of the solver's residual norm, per output step.

    This is the reduction that replaces storing the `resid_<var>` fields: seven
    numbers per step instead of seven full fields, and it is what the analysis
    reads anyway. Definition follows the residual-norm notebook:

        SS_v(t)    = sum over interior cells of resid_v(cell, t)**2
        share_v(t) = SS_v(t) / sum over equations of SS_w(t)

    No volume weighting -- the solver's norm is a plain sum over state-vector
    entries, so weighting by cell volume would measure something the solver does
    not feel. Shares rather than magnitudes, because raw residual magnitude
    scales with the adaptive timestep and cannot be compared across time or
    between runs, while a share is dimensionless and directly comparable.
    """

    import numpy as np
    import pandas as pd

    fields = _resid_fields(ds)
    if not fields:
        return None

    mask = _interior_mask(ds)
    squares = {}
    for name in fields:
        field = _interior(ds[name], mask)
        dims = [d for d in field.dims if d != "t"]
        squares[str(name)[len("resid_"):]] = (field ** 2).sum(dims).values

    total = np.sum(list(squares.values()), axis=0)
    out = {"resid_ss_total": total}
    with np.errstate(invalid="ignore", divide="ignore"):
        for equation, ss in squares.items():
            out[f"share_{equation}"] = np.where(total > 0, ss / total, np.nan)

    return pd.DataFrame(out)


def _physics_series(ds, report=None):
    """
    The physics monitor quantities, per output step.

    The same four the live monitor and the physics-build reports draw, so a
    solver report can be checked against them: density and temperature at the
    outer midplane separatrix, and the maxima at the targets.

    Two definitions differ from the monitor's on purpose. The target maxima use
    the store's own region selector, the one behind the `ne_target_max` and
    `te_target_max` columns, so the history and the recorded endpoint are the
    same quantity -- 0.2% from the monitor's index arithmetic, which averages a
    guard cell this dataset does not carry. And the separatrix columns are named
    for the midplane they use: this is a connected double null with four of
    them, and the index deliberately records no unqualified `ne_sep`.
    """

    import numpy as np
    import pandas as pd

    if "t" not in ds:
        return None

    meta = dict(ds.metadata) if hasattr(ds, "metadata") else {}
    columns = {"t": np.atleast_1d(ds["t"].values)}
    try:
        j1_2, j2_2 = meta["jyseps1_2g"], meta["jyseps2_2g"]
        y_omp = int((j2_2 - j1_2) / 2) + j1_2
        x_sep = meta["ixseps1"]
        for column, field in (("ne_omp_sep", "Ne"), ("te_omp_sep", "Te")):
            if field in ds:
                columns[column] = ds[field].isel(x=x_sep, theta=y_omp).values
    except Exception as exc:  # noqa: BLE001 - one panel must not lose the rest
        if report is not None:
            report.warnings.append(f"midplane series not extracted: {exc}")

    try:
        targets = ds.hermes.select_region(radial_region="domain",
                                          poloidal_region="targets")
        dims = [d for d in targets["Te"].dims if d != "t"] if "Te" in targets else []
        for column, field in (("ne_target_max", "Ne"), ("te_target_max", "Te")):
            if field in targets:
                columns[column] = targets[field].max(dims).values
    except Exception as exc:  # noqa: BLE001
        if report is not None:
            report.warnings.append(f"target series not extracted: {exc}")

    if len(columns) == 1:
        return None
    return pd.DataFrame(columns)


def _ddt_series(ds):
    """
    Root-mean-square time derivative per evolved variable, per output step.

    How fast the solution is still moving, which is the question "has this run
    reached a steady state" and is not answerable from the residual: the
    residual says how well the implicit step was solved, this says how much the
    step changed.

    Two norms, because two tools disagree on purpose. `rms_ddt` is a plain RMS
    over interior cells, matching the way the solver's own norms are taken, so
    a big cell does not count for more than a small one. `rms_ddt_vw` weights by
    cell volume (dx dy dz J), which is what cmonitor plots live during a run --
    recorded so a report can be laid beside the monitor a person was watching.

    `conversion` is the SI value of one normalised unit for that variable, so
    either norm can be put back into the normalised units cmonitor shows.

    `max_abs_ddt` travels beside the RMS because they answer different
    questions: an RMS over 3600 cells dilutes a spike in a handful of them by a
    factor of sixty, so a run whose RMS is flat can still be fluctuating hard
    somewhere. Check the two together before concluding a run is quiet.

    `rms_state` travels beside it so a fractional rate can be formed as
    rms_ddt / rms_state, in units of 1/s. The pointwise alternative, RMS of
    ddt(v)/v cell by cell, is deliberately NOT recorded: momentum changes sign,
    so it divides by values passing through zero and returns 3e4 where the
    ratio of norms returns 69.

    Guard cells are dropped for the same reason as the residual fields (F20).
    """

    import numpy as np
    import pandas as pd

    if "t" not in ds:
        return None
    fields = sorted(v for v in ds.data_vars if str(v).startswith("ddt("))
    if not fields:
        return None

    mask = _interior_mask(ds)
    times = np.atleast_1d(ds["t"].values)

    volume = None
    if all(n in ds for n in ("dx", "dy", "dz", "J")):
        volume = _interior(ds["dx"] * ds["dy"] * ds["dz"] * ds["J"], mask)

    rows = []
    for name in fields:
        variable = str(name)[len("ddt("):-1]
        rate = _interior(ds[name], mask)
        dims = [d for d in rate.dims if d != "t"]
        rms_ddt = np.sqrt((rate ** 2).mean(dims).values)
        max_abs = np.abs(rate).max(dims).values
        if volume is None:
            rms_vw = np.full(len(times), np.nan)
        else:
            rms_vw = np.sqrt(((rate ** 2) * volume).sum(dims).values
                             / float(volume.sum()))
        conversion = float(ds[name].attrs.get("conversion", np.nan))
        if variable in ds:
            state = _interior(ds[variable], mask)
            rms_state = np.sqrt((state ** 2).mean(dims).values)
        else:
            rms_state = np.full(len(times), np.nan)
        for i, t in enumerate(times):
            rows.append((t, variable, rms_ddt[i], rms_vw[i], max_abs[i],
                         rms_state[i], conversion))

    return pd.DataFrame(rows, columns=["t", "variable", "rms_ddt",
                                       "rms_ddt_vw", "max_abs_ddt",
                                       "rms_state", "conversion"])


def _residual_regions(ds, report=None):
    """
    Per-region, per-equation share of the residual norm, per output step.

    The shares in `series.tsv` say which equation the solver is waiting for;
    these say where in the machine it is waiting. Long format -- one row per
    (step, region, equation) -- because the region set is a property of the
    topology and a wide table would change shape between grids.

    `share` is against the whole-domain total at that step, so it is the same
    denominator as `share_<equation>` in the series and the two tables can be
    read together. Summing every share at one step gives the fraction the
    region set accounted for; whatever is missing is recorded per step as the
    `unassigned` region rather than left to be inferred.

    A region is a plain sum of squares, so a large region can score highly by
    holding more cells. The cell count travels with each row for exactly that
    reason.

    Stored sparsely: a (step, region, equation) row whose share is under
    RESID_REGION_FLOOR is dropped and MUST be read as zero. Almost every
    combination is that small -- keeping them multiplies the table by thirty and
    decides nothing at one part in a million of the norm. The `unassigned` rows
    are always kept, since a remainder is only useful if it is always there.
    """

    import numpy as np
    import pandas as pd

    fields = _resid_fields(ds)
    if not fields or not hasattr(ds[fields[0]], "hermes"):
        return None

    mask = _interior_mask(ds, report)
    times = np.atleast_1d(ds["t"].values) if "t" in ds else None
    if times is None:
        return None

    squares, covered = {}, None
    total = np.zeros(len(times))
    for name in fields:
        field = _interior(ds[name], mask)
        dims = [d for d in field.dims if d != "t"]
        total = total + (field ** 2).sum(dims).values

    rows, cells = [], {}
    for label, (radial, poloidal) in RESID_REGIONS.items():
        for name in fields:
            try:
                sub = ds[name].hermes.select_region(radial_region=radial,
                                                    poloidal_region=poloidal)
            except Exception as e:  # noqa: BLE001 -- a grid without this region
                if report is not None:
                    report.warnings.append(
                        f"residual region {label} unavailable: {e}")
                sub = None
            if sub is None:
                continue
            dims = [d for d in sub.dims if d != "t"]
            ss = (sub ** 2).sum(dims).values
            cells[label] = int(np.prod([sub.sizes[d] for d in dims]))
            squares[(label, str(name)[len("resid_"):])] = ss
            covered = ss if covered is None else covered + ss

    if not squares:
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        for (label, equation), ss in squares.items():
            share = np.where(total > 0, ss / total, np.nan)
            for i, t in enumerate(times):
                rows.append((t, label, equation, cells[label], share[i]))
    frame = pd.DataFrame(rows, columns=["t", "region", "equation", "cells",
                                        "share"])
    frame = frame[frame["share"].abs() > RESID_REGION_FLOOR]

    with np.errstate(invalid="ignore", divide="ignore"):
        remainder = np.where(total > 0, 1.0 - covered / total, np.nan)
    tail = pd.DataFrame({"t": times, "region": "unassigned", "equation": "all",
                         "cells": 0, "share": remainder})

    return pd.concat([frame, tail], ignore_index=True)


def _dump_series(ds):
    """
    The per-output-step history, as a table.

    Without this the bundle holds only end-of-run scalars, and how a run got
    there -- where it slowed down, when the residual stalled -- dies with the
    dumps. Solver-specific fields are included when present and skipped when
    not, so the same function serves SNES and CVODE.
    """

    import numpy as np
    import pandas as pd

    if "t" not in ds:
        return None

    columns = {"t": np.atleast_1d(ds["t"].values)}
    length = len(columns["t"])
    for name in SERIES_FIELDS:
        if name not in ds:
            continue
        values = np.atleast_1d(ds[name].values)
        if values.ndim == 1 and values.size == length:
            columns[name] = values

    frame = pd.DataFrame(columns)

    shares = _residual_shares(ds)
    if shares is not None and len(shares) == len(frame):
        frame = pd.concat([frame, shares], axis=1)
    return frame


def _residual_metrics(ds, ncalls):
    """
    Convergence, reduced to three scalars. SNES only -- CVODE writes no residual.

    Compared within one `scale_vars` setting only: the norm is an unweighted RMS
    in normalised units and the weights are recorded nowhere, so a residual
    measured under scaling cannot be un-weighted afterwards.
    """

    if "snes_global_residual" not in ds:
        return {}

    import numpy as np

    resid = np.asarray(ds["snes_global_residual"].values, dtype=float)
    resid = resid[np.isfinite(resid) & (resid > 0)]
    if resid.size < 2:
        return {}

    out = {
        "resid_final": float(resid[-1]),
        "resid_drop": float(np.log10(resid[0] / resid[-1])),
    }
    if ncalls:
        out["resid_per_rhs"] = out["resid_drop"] / ncalls
    return out


def _physics_gates(ds, report):
    """
    The correctness quantities, in SI, at the end of the test.

    Only the target maxima are computed. A separatrix value needs a poloidal
    location to be meaningful and this is a connected double null with four
    midplanes, so `ne_sep`/`te_sep` wait for that convention rather than being
    filled with an arbitrary choice.
    """

    import numpy as np

    out = {}
    try:
        # Full radial extent at the targets: `domain` rather than `sol`, so a
        # peak that sits inside the separatrix is not silently excluded.
        targets = ds.hermes.select_region(
            radial_region="domain", poloidal_region="targets"
        )
        for column, field in (("ne_target_max", "Ne"), ("te_target_max", "Te")):
            if field not in targets:
                continue
            values = targets[field].isel(t=-1).values
            if np.isfinite(values).any():
                out[column] = float(np.nanmax(values))
    except Exception as exc:  # noqa: BLE001 - one missing metric must not lose a run
        report.warnings.append(f"target quantities not extracted: {exc}")

    report.warnings.append(
        "ne_sep/te_sep left empty: no separatrix location convention is agreed"
    )
    return out


# =============================================================================
# The bundle
# =============================================================================
def _write_bundle(case_dir, bundle_dir, tables, notes, report):
    """Copy the evidence and write the parsed tables beside it."""

    os.makedirs(bundle_dir, exist_ok=True)

    from hermes3 import logparse as lp

    for name in BUNDLE_FILES:
        src = os.path.join(case_dir, name)
        if os.path.exists(src) and name == lp.CONSOLE_LOG:
            # Trimmed, not copied: the repeated per-solve object dumps are most
            # of the file and none of the information.
            text, dropped = lp.trim_console(src)
            with open(os.path.join(bundle_dir, name), "w") as f:
                f.write(text)
            if dropped:
                report.warnings.append(
                    f"{dropped} repeated PETSc object dump(s) trimmed from the"
                    " bundled console"
                )
        elif os.path.exists(src):
            shutil.copy2(src, os.path.join(bundle_dir, name))
        elif name == "BOUT.log.console":
            report.warnings.append(
                "no BOUT.log.console: launched without teeing stdout, so the"
                " log_view report, options_left and the PETSc version are lost"
            )
        else:
            report.problems.append(f"missing {name}")

    for name, frame in tables.items():
        if frame is None or frame.empty:
            continue
        frame.to_csv(
            os.path.join(bundle_dir, name),
            sep="\t",
            float_format=FLOAT_FORMAT,
            index=frame.index.name is not None,
        )

    if notes:
        with open(os.path.join(bundle_dir, "petsc_views.txt"), "w") as f:
            f.write(notes)


# =============================================================================
# Entry point
# =============================================================================
def extract_case(
    case_dir,
    store_dir,
    grid_path=None,
    index_name="index.tsv",
    recipes_dir=None,
    conduction_method=None,
    epoch=None,
    dry_run=False,
):
    """
    Extract one finished case into `store_dir`, and say whether it worked.

    Writes `store_dir/runs/<test_id>/` and fills the open index row for this
    case directory. A finished run with no open row is recorded as `unplanned`
    rather than refused: losing a result to enforce process is a bad trade.

    Returns a Report. `report.ok` is the answer to "may I delete the dumps".
    """

    from hermes3 import logparse as lp

    report = Report(case_dir)
    series = None
    resid_regions = None
    ddt = None
    physics = None
    if not os.path.isdir(case_dir):
        report.problems.append("case directory does not exist")
        return report
    if not os.path.exists(os.path.join(case_dir, "BOUT.log.0")):
        report.problems.append("no BOUT.log.0 - nothing to extract")
        return report

    # --- text ------------------------------------------------------------
    header = lp.run_header(case_dir)
    steps = lp.output_steps(case_dir)
    snes = lp.snes_steps(case_dir)

    events = None
    console = os.path.join(case_dir, lp.CONSOLE_LOG)
    notes = ""
    measured = {}
    if os.path.exists(console):
        measured["petsc_version"] = lp.petsc_version(console)
        banner = lp.run_banner(console)
        measured.update(
            {k: v for k, v in banner.items() if k in ("slot", "hermes_branch")}
        )
        try:
            events = lp.parse_petsc_logview(console)
        except ValueError as exc:
            report.warnings.append(str(exc).split(".")[0])

        unused = lp.options_left(console)
        if unused:
            report.warnings.append(
                "PETSc never used these options: " + ", ".join(unused)
            )
        notes = "\n\n".join(
            block
            for block in (
                f"PETSc version: {measured.get('petsc_version')}",
                f"options_left: {unused}",
                lp.first_view_block(console, "SNES Object"),
                lp.first_view_block(console, "KSP Object"),
            )
            if block
        )

    # --- what happened ---------------------------------------------------
    finished = _settings_finished(case_dir)
    nout = _inp_value(case_dir, "", "nout") or _inp_value(case_dir, "run", "nout")
    expected = int(nout) + 1 if nout and nout.isdigit() else None

    if header["snes_failed"]:
        measured["outcome"] = "snes_failure"
    elif not finished:
        report.warnings.append(
            "BOUT.settings is the startup stub: the run was killed or is still"
            " going. outcome left for a human call"
        )
    elif expected and len(steps) == expected:
        measured["outcome"] = "completed"
    else:
        report.warnings.append(
            f"finished but wrote {len(steps)} of {expected} output steps;"
            " outcome left for a human call"
        )

    measured["wall_s"] = header["wall_s"]
    measured["run_started"] = header["run_started"]
    for key in ("bout_version", "bout_commit", "hermes_commit", "check_level",
                "build_type"):
        measured[key] = header[key]

    if not snes.empty:
        measured["nl_its"] = int(snes["nl_its"].sum())
        measured["lin_its"] = int(snes["lin_its"].sum())
        measured["solver_fails"] = int(snes["solver_fails"].max())

    if events is not None:
        total = events["time_max"].get("Total BOUT++")
        for column, event in TIME_SHARES.items():
            if event in events.index and total:
                measured[column] = float(events["time_max"][event] / total)
        if "SNESJacobianEval" in events.index:
            measured["n_jac_builds"] = int(events["count_max"]["SNESJacobianEval"])

    # --- the dumps -------------------------------------------------------
    grid = _resolve_grid(case_dir, grid_path)
    if grid is None:
        report.problems.append(
            "grid file not found - pass grid_path; without the dumps there are no"
            " solver counters, no provenance constants and no physics numbers"
        )
    else:
        measured["grid"] = os.path.basename(grid)
        try:
            from_dump = _read_dump(case_dir, grid, report)
            series = from_dump.pop("_series", None)
            resid_regions = from_dump.pop("_resid_regions", None)
            ddt = from_dump.pop("_ddt", None)
            physics = from_dump.pop("_physics", None)
            shape = from_dump.pop("grid_shape", None)
            if shape:
                measured["grid"] = f"{os.path.basename(grid)} ({shape})"
            from_dump.pop("n_output_steps", None)
            measured.update(from_dump)
        except Exception as exc:  # noqa: BLE001
            report.problems.append(f"could not read the dumps: {exc}")

    if measured.get("wall_s") and measured.get("sim_time_ms"):
        # Simulated milliseconds per 24 hours of wall clock. Stated here because
        # sdtools holds several implementations of "speed" that disagree.
        measured["ms_per_24h"] = (
            measured["sim_time_ms"] / measured["wall_s"] * 86400.0
        )

    # --- identity --------------------------------------------------------
    test = _infer_test(case_dir)
    if header["started_at"]:
        report.test_id = f"{test}-{header['started_at']:%Y%m%d-%H%M%S}"
    else:
        report.problems.append("no run start time in BOUT.log.0 - cannot form test_id")

    measured.update(
        {
            "test_id": report.test_id,
            "case_dir": idx.case_key(case_dir),
            "test": test,
            "conduction_method": conduction_method,
            "epoch": epoch,
            "machine": socket.gethostname(),
            "originator": getpass.getuser(),
            "recorded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "verdict": "no_reference",
        }
    )
    measured = {k: v for k, v in measured.items() if v is not None and v != ""}
    report.record = measured

    index_path = os.path.join(store_dir, index_name)
    rows, columns = idx.read_index(index_path)

    # test_id is <test>-<start time to the second>, which is NOT unique: runs
    # launched into different slots routinely start in the same second. Left
    # alone, the second one silently overwrites the first one's bundle.
    if report.test_id:
        key = idx.case_key(case_dir)
        taken = {
            r.get("test_id")
            for r in rows
            if r.get("test_id") and idx.case_key(r.get("case_dir", "")) != key
        }
        if report.test_id in taken:
            stem, suffix = report.test_id, 2
            while f"{stem}-{suffix}" in taken:
                suffix += 1
            report.test_id = f"{stem}-{suffix}"
            report.warnings.append(
                f"another run of this test started in the same second; test_id"
                f" disambiguated to {report.test_id}"
            )
        measured["test_id"] = report.test_id

    if dry_run:
        return report

    # --- write -----------------------------------------------------------
    if report.test_id:
        report.bundle = os.path.join(store_dir, "runs", report.test_id)
        _write_bundle(
            case_dir,
            report.bundle,
            {
                "steps.tsv": steps,
                "snes_steps.tsv": snes,
                "events.tsv": events,
                "series.tsv": series,
                "resid_regions.tsv": resid_regions,
                "ddt.tsv": ddt,
                "physics.tsv": physics,
            },
            notes,
            report,
        )

    position = idx.find_open_row(rows, case_dir)

    if position is None:
        existing = [
            i
            for i, r in enumerate(rows)
            if idx.case_key(r.get("case_dir", "")) == idx.case_key(case_dir)
            and r.get("state") in (idx.STATE_RECORDED, idx.STATE_UNPLANNED)
        ]
        if existing:
            position = existing[0]
            report.warnings.append(
                "row already recorded for this test_id: re-extracted in place"
            )

    if position is None:
        report.warnings.append("no open row for this case directory: recorded as unplanned")
        row = {c: "" for c in columns}
        row["state"] = idx.STATE_UNPLANNED
        rows.append(row)
    else:
        row = rows[position]
        row["state"] = idx.STATE_RECORDED

    # The recipe is declared, so the diff against it can only be computed once
    # the row is in hand.
    named = row.get("recipe", "") or ""
    recipe_path = recipe.find_recipe(named, recipes_dir)
    if named and recipe_path is None:
        report.warnings.append(f"recipe {named!r} not found: diffs not computed")
    elif recipe_path:
        diffs = recipe.diff_against_recipe(case_dir, recipe_path)
        measured["diffs"] = "; ".join(diffs)
        _check_varied(row.get("varied", ""), diffs, report)

    filled, conflicts, unknown = idx.fill_row(row, measured, columns)
    report.filled = filled
    report.conflicts = conflicts
    if unknown:
        report.problems.append(
            "measured values with no column in the index: " + ", ".join(unknown)
        )
    if conflicts:
        report.warnings.append(
            "declared values kept, measurements not written for: "
            + ", ".join(conflicts)
        )

    moved = idx.recompute_concurrency(rows)
    if moved:
        report.warnings.append(
            f"concurrency recomputed on {moved} row(s): adding this run changes what"
            " other runs overlapped"
        )

    idx.write_index(index_path, rows, columns)
    return report


def _check_varied(varied, diffs, report):
    """
    Flag settings that differ from the recipe but were not declared as the thing
    being tested. A deviation nobody declared is a stale input file or a drifted
    recipe, and it invalidates the comparison rather than adding to it.
    """

    declared = {
        part.split("=")[0].strip() for part in varied.split(";") if part.strip()
    }
    undeclared = [
        d for d in diffs if d.split(":")[0] + ":" + d.split(":")[1] not in declared
    ]
    if varied and undeclared:
        report.warnings.append(
            f"{len(undeclared)} setting(s) differ from the recipe but are not in"
            f" `varied`, first: {undeclared[0]}"
        )
