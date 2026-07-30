"""Scan pages: several scans compared as lines against the sweep coordinate.

These take a SeriesSet (ctx.series), not a CaseSet. Every case is already
reduced to scalars by SeriesSet.load(), so these pages only ever plot numbers.
"""

import textwrap
from inspect import cleandoc

import matplotlib.pyplot as plt

from . import register_page
from .. import provenance
from ..report import raw_page, text_table, timestamp, priority_note

# Text scale for the plot pages. Bump this to grow every text element (titles,
# tick labels, axis labels, legend) together; the sizes below are base x scale,
# applied via an rc_context so hardcoded per-artist sizes don't fight it.
FONT_SCALE = 1.3
PLOT_RC = {
    "font.size": 10 * FONT_SCALE,
    "axes.titlesize": 11 * FONT_SCALE,
    "axes.labelsize": 11 * FONT_SCALE,
    "xtick.labelsize": 10 * FONT_SCALE,
    "ytick.labelsize": 10 * FONT_SCALE,
    "legend.fontsize": 10 * FONT_SCALE,
    "legend.title_fontsize": 11 * FONT_SCALE,
    "figure.titlesize": 15 * FONT_SCALE,
}


def _grid(ax):
    ax.minorticks_on()
    ax.grid(which="major", color="0.55", alpha=0.63, lw=0.9)
    ax.grid(which="minor", color="0.7", alpha=0.49, lw=0.6, ls=":")
    ax.set_axisbelow(True)


def _scan_axis(ax, series, points, yscale, title):
    """One scan panel: y vs the sweep coordinate, one line per series."""
    for label in series.labels:
        xy = points.get(label, [])
        if not xy:
            continue
        # No marker edge (edge defaults to the face colour); markers sized up so
        # a series whose line coincides with another still shows through.
        ax.plot([p[0] for p in xy], [p[1] for p in xy],
                marker=series.marker(label), ls="-", color=series.color(label),
                lw=1.8, ms=10.5, alpha=0.9, label=label)
    ax.set_yscale(yscale)
    ax.set_xlabel(series.x_label)
    ax.set_title(title)  # size from axes.titlesize (see PLOT_RC)
    _grid(ax)


def _panel_grid(ctx, n_panels):
    """A 2x3 page of panels with the spare axes freed for the legend."""
    fig, axs = plt.subplots(2, 3, figsize=ctx.page_size, dpi=110)
    axs = axs.ravel()
    for ax in axs[n_panels:]:
        ax.set_axis_off()
    return fig, axs


def _shared_legend(fig, axs, n_panels):
    """Put one legend in the first spare axes, or below the row if there is
    none (loc='best' inside a panel always lands on the data)."""
    handles, labels = axs[0].get_legend_handles_labels()
    if not handles:
        return
    if n_panels < len(axs):
        axs[-1].legend(handles, labels, loc="center", title="Series",
                       frameon=False)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))


# =============================================================================
@register_page("scan_cover")
def scan_cover_page(ctx, options=(), fontsize=6.5, wrap=110):
    """Provenance cover for a scan study: series summary, per-case provenance,
    and a series-vs-option table of every option that differs across series.

    options : DISPLAY-ORDER HINT ONLY -- never a filter (it used to be one, and
        silently hid renamed knobs; see provenance.diff_option_sets). Entries
        match a flattened-key prefix ("solver:") or a bare option name
        ("flux_limit" matches "d:flux_limit"). Defaults to the campaign's
        param_diff_priority.
    """
    series = ctx.series
    camp = ctx.campaign

    L = ["ANALYSIS ENVIRONMENT (at PDF build time)"]
    width = max((len(k) for k in camp.repos), default=0)
    for label, path in camp.repos.items():
        L.append(f"  {label.ljust(width)} : {provenance.git_describe(path)}")
    L.append("")

    L.append("CONCLUSIONS")
    for para in cleandoc(ctx.notes or "(none recorded)").splitlines():
        L += ["  " + wl for wl in (textwrap.wrap(para, wrap) or [""])]
    L.append("")

    L.append("SERIES")
    srows = []
    for label in series.labels:
        glob = series.specs[label]["glob"]
        glob_str = glob if isinstance(glob, str) else " + ".join(glob)
        xs = [f"{x:g}" for _, x in series.matched.get(label, [])]
        srows.append([label, glob_str, str(len(xs)),
                      ", ".join(xs) or "(none found)"])
    L += text_table(["series", "glob", "n", f"{series.x_name} points"],
                    srows).splitlines()
    L.append("")

    L.append("CASES (as run)")
    crows = []
    for label in series.labels:
        for name, x in series.matched.get(label, []):
            d = series.loader.dir(name)
            ri = provenance.run_info(d)
            crows.append([label, f"{x:g}", name, ri["date"], ri["hermes"],
                          f"CHK{provenance.check_level(d)}"])
    if crows:
        L += text_table(
            ["series", series.x_name, "sim id", "run date", "hermes", "check"],
            crows,
        ).splitlines()
    L.append("")

    optvals = series.option_values()
    if optvals:
        # The SAME core the case covers use: compares EVERY option of every
        # case, with the campaign's key list only ORDERING the result.
        priority = tuple(options) or camp.param_diff_priority
        differing, per, _ = provenance.diff_option_sets(optvals, priority=priority)
        labels = list(optvals)
        order = priority_note(priority)
        L.append(f"DIFFERING OPTIONS  (source: BOUT.log.0, i.e. what each run "
                 f"actually read; all sections; run-provenance excluded; "
                 f"{order}; '{provenance.VARIES}' = not constant within "
                 f"the series)")
        if differing:
            def clip(v):
                return v if len(v) <= 40 else v[:39] + "…"
            L += text_table(
                ["option", *labels],
                [[k, *(clip(per[l][k]) for l in labels)] for k in differing],
            ).splitlines()
        else:
            L.append("  (every option identical across series)")

    fig = plt.figure(figsize=ctx.page_size)
    fig.text(0.06, 0.97, ctx.slug, ha="left", va="top", fontsize=15,
             fontweight="bold")
    fig.text(0.06, 0.945,
             f"campaign: {camp.name}    built {timestamp()}",
             ha="left", va="top", fontsize=8, color="0.35")
    fig.text(0.06, 0.91, "\n".join(L), ha="left", va="top",
             family="monospace", fontsize=fontsize)
    return raw_page(fig)


@register_page("scan_scalars")
def scan_scalars_page(ctx, scalars=(), title="Physics vs scan parameter"):
    """One panel per physics scalar: scalar vs the sweep coordinate.

    scalars : the campaign's scalar specs (the same list SeriesSet reduces
        with), each carrying `key`, `title` and `yscale`.
    """
    series = ctx.series
    specs = list(scalars)
    with plt.rc_context({**PLOT_RC, "savefig.bbox": None}):
        fig, axs = _panel_grid(ctx, len(specs))
        for ax, s in zip(axs, specs):
            _scan_axis(ax, series, series.scalar_points(s["key"]),
                       s.get("yscale", "linear"), s["title"])
        _shared_legend(fig, axs, len(specs))
        fig.suptitle(title, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return raw_page(fig, rc=PLOT_RC)


@register_page("scan_performance")
def scan_performance_page(ctx, title="Performance vs scan parameter"):
    """Mean speed, total runtime and speed variability vs the sweep coordinate.

    Total runtime is the plainest cost scalar; the variability metric is set on
    the campaign (see scalars.perf_scalars) and its label follows automatically.
    """
    from ..scalars import VARIABILITY_LABELS

    series = ctx.series
    with plt.rc_context({**PLOT_RC, "savefig.bbox": None}):
        fig, axs = plt.subplots(
            1, 3, figsize=(ctx.page_size[0], ctx.page_size[1] * 0.55), dpi=110
        )
        _scan_axis(axs[0], series, series.perf_points("mean_speed"), "linear",
                   "Mean speed  [ms sim / 24 hr wall]")
        _scan_axis(axs[1], series, series.perf_points("total_wall_hr"), "linear",
                   "Total runtime  [hr]")
        _scan_axis(axs[2], series, series.perf_points("variability"), "linear",
                   VARIABILITY_LABELS.get(series.perf_metric, "Speed variability"))

        handles, labels = axs[0].get_legend_handles_labels()
        fig.suptitle(title, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0.16, 1, 0.9))
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=3,
                       frameon=False, bbox_to_anchor=(0.5, 0.0))
    return raw_page(fig, rc=PLOT_RC)


@register_page("scan_tradeoff")
def scan_tradeoff_page(ctx, scalars=(), perf_key="mean_speed",
                       perf_label="Mean speed  [ms sim / 24 hr wall]"):
    """Speed (y) against each physics scalar (x), one panel per scalar.

    Reads the speed/accuracy TRADE directly: series whose points sit up-and-left
    (faster at the same or smaller answer shift) win; a series that is faster
    only where its physics scalar has moved bought its speed with a state
    change. Points are connected in sweep order and labelled with the sweep
    value, since that is no longer an axis.
    """
    series = ctx.series
    specs = list(scalars)
    with plt.rc_context({**PLOT_RC, "savefig.bbox": None}):
        fig, axs = _panel_grid(ctx, len(specs))
        for ax, s in zip(axs, specs):
            pts_by_label = series.paired_points(s["key"], perf_key)
            for label in series.labels:
                pts = pts_by_label.get(label, [])
                if not pts:
                    continue
                colour = series.color(label)
                ax.plot([p[1] for p in pts], [p[2] for p in pts],
                        marker=series.marker(label), ls="-", color=colour,
                        lw=1.6, ms=9, alpha=0.85, label=label)
                for xv, x, y in pts:
                    ax.annotate(f"{xv:g}", (x, y), textcoords="offset points",
                                xytext=(5, 5), fontsize=6.5 * FONT_SCALE,
                                color=colour, alpha=0.95)
            ax.set_xscale(s.get("yscale", "linear"))
            ax.set_xlabel(s["title"].replace("\n", " "),
                          fontsize=8.5 * FONT_SCALE)
            ax.set_ylabel(perf_label, fontsize=8.5 * FONT_SCALE)
            _grid(ax)
        _shared_legend(fig, axs, len(specs))
        fig.suptitle(f"Performance vs physics  (point labels = "
                     f"{series.x_name.lower()})", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return raw_page(fig, rc=PLOT_RC)
