"""Reducing one case to a few numbers.

A scan campaign plots scalars against a scan coordinate, so every case must
collapse to a handful of values. Two families live here:

  * performance scalars -- speed and stutter, derived from t/wtime alone, so
    they are the same for every campaign,
  * profile scalars -- one value pulled from a 1-D profile, where WHICH value
    is editorial. The reduction vocabulary is general; the campaign declares
    the (variable, region, reduction) triples it cares about.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Profile reductions. `Srad` is the radial coordinate relative to the
# separatrix (negative = core side) that the sdtools selectors attach.
# ---------------------------------------------------------------------------
REDUCTIONS = {
    # innermost domain cell of a radial profile -- the core-edge value
    "inner": lambda df, var: float(df.loc[df["Srad"].idxmin(), var]),
    # outermost domain cell -- the far-SOL / wall-edge value
    "outer": lambda df, var: float(df.loc[df["Srad"].idxmax(), var]),
    # first ring outside the separatrix (sepadd = 1)
    "sol_ring1": lambda df, var: float(
        df[df["Srad"] > 0].sort_values("Srad").iloc[0][var]
    ),
    "absmax": lambda df, var: float(df[var].abs().max()),
    "max": lambda df, var: float(df[var].max()),
    "min": lambda df, var: float(df[var].min()),
    "mean": lambda df, var: float(df[var].mean()),
}


def profile_scalars(rec, specs):
    """Reduce one case record to {key: value} per spec.

    specs : sequence of dicts, each
        key    -- name of the resulting scalar
        var    -- Hermes-3 variable, e.g. "Ne" (the RAW dataset name, not a
                  plot-label alias like "Na" that comparison tools invent)
        region -- named selection, e.g. "omp" or "outer_lower_target"
        reduce -- a key of REDUCTIONS

    Specs sharing a region are evaluated from ONE selector call, since pulling
    a radial profile is the expensive part. A value that can't be computed
    comes back NaN so a scan line simply skips that point rather than breaking.
    """
    from hermes3.selectors import get_1d_radial_data

    out = {s["key"]: np.nan for s in specs}

    by_region = {}
    for s in specs:
        by_region.setdefault(s["region"], []).append(s)

    for region, group in by_region.items():
        variables = sorted({s["var"] for s in group})
        try:
            df = get_1d_radial_data(rec["last"], variables, region=region)
        except Exception as e:  # noqa: BLE001
            print(f"    [scalars: region {region}] {e}")
            continue
        for s in group:
            if s["var"] not in df:
                continue
            try:
                out[s["key"]] = REDUCTIONS[s["reduce"]](df, s["var"])
            except Exception as e:  # noqa: BLE001
                print(f"    [scalars: {s['key']}] {e}")
    return out


# ---------------------------------------------------------------------------
# Performance scalars
# ---------------------------------------------------------------------------
VARIABILITY_LABELS = {
    "p95_over_median": "Speed variability\np95 / median step wall time",
    "stall_fraction": "Stall fraction\nwall time in steps > 2x median",
    "dip_factor": "Worst dip factor\nmax / min smoothed speed",
}


def perf_scalars(rec, metric="p95_over_median"):
    """Mean speed, total runtime and a speed-variability scalar for one record.

    mean_speed : ms of simulation time per 24 h of wall time, over the whole
        run (final_sim_ms / (total_wall_hr / 24)).

    variability : one number capturing how much the run stutters. Hermes writes
        output at fixed sim-time steps; each interval takes however much wall
        time the solver needs, so when the internal timestep collapses that
        interval's `wtime` spikes then recovers. A steady run has near-constant
        wtime; a stuttering run has spikes. `metric` selects the scalar:
          p95_over_median : p95(wtime) / median(wtime); 1 = steady.
          stall_fraction  : share of wall time in intervals > 2x median.
          dip_factor      : max / min of the 20-step rolling speed.
    """
    t_ms, wtime = rec["t_ms"], rec["wtime"]
    final_ms = float(t_ms[-1]) if t_ms.size else np.nan
    total_wall_hr = float(np.nansum(wtime)) / 3600.0

    mean_speed = np.nan
    if total_wall_hr > 0 and np.isfinite(final_ms):
        mean_speed = final_ms / (total_wall_hr / 24.0)

    # Drop the first couple of outputs: the initial interval carries one-off
    # setup cost that is not part of the steady stutter we want to measure.
    w = wtime[2:] if wtime.size > 4 else wtime
    w = w[np.isfinite(w) & (w > 0)]

    variability = np.nan
    if w.size:
        med = np.median(w)
        if metric == "p95_over_median" and med > 0:
            variability = float(np.percentile(w, 95) / med)
        elif metric == "stall_fraction":
            variability = float(w[w > 2 * med].sum() / w.sum())
        elif metric == "dip_factor":
            # 20-step rolling mean of the per-interval speed proxy (1/wtime).
            speed = 1.0 / w
            k = min(20, speed.size)
            ma = np.convolve(speed, np.ones(k) / k, mode="valid")
            ma = ma[ma > 0]
            if ma.size:
                variability = float(ma.max() / ma.min())

    return dict(mean_speed=mean_speed, variability=variability,
                final_ms=final_ms, total_wall_hr=total_wall_hr)
