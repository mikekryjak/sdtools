"""Physics monitor page: the cmonitor.py top-row quantities, cases overlaid."""

from . import register_page


@register_page("monitor")
def monitor_page(ctx, neutrals=False, species="d", dpi=110):
    """Four time-history panels -- the cmonitor.py top row -- one line per case.

    neutrals=False is cmonitor's default row (Ne/Te at the OMP separatrix and
    the target maximum); True is its -neutrals variant (neutral Tn/Nn and the
    core-average Tn) for neutral species tag `species`.
    """
    from hermes3.plotting import compare_monitor_physics

    return compare_monitor_physics(
        ctx.cases.datasets(),
        colors=ctx.cases.colors(),
        neutrals=neutrals,
        species=species,
        dpi=dpi,
    )
