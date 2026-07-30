"""Runtime breakdown page: each case reduced to one wall-time bar."""

from . import register_page


@register_page("runtime_breakdown")
def runtime_breakdown_page(ctx, n_phases=5, endpoint_ms=None, dpi=110):
    """One stacked horizontal bar per case: wall hours to a COMMON sim-time
    endpoint (default the shortest final sim-time in the set), split into
    `n_phases` equal sim-time bands so a mid-run slowdown shows as a fat
    segment.

    Bars are annotated with the CHECK level, and a set mixing CHECK levels is
    flagged -- wall times are only comparable at equal CHECK and core count.
    """
    from hermes3.plotting import plot_runtime_breakdown

    checks = ctx.cases.check_levels()
    entries = {
        ctx.cases.label(n): dict(
            ds=ctx.cases.loader.ds(n),
            color=ctx.cases.color(n),
            check=checks[n],
        )
        for n in ctx.cases.names
    }
    return plot_runtime_breakdown(entries, n_phases=n_phases,
                                  endpoint_ms=endpoint_ms, dpi=dpi)
