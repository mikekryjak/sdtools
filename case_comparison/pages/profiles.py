"""Profile pages: 1D profiles at the last timestep, cases overlaid."""

from . import register_page


@register_page("profiles")
def profiles_page(ctx, params=("Td+", "Te", "Td", "Ne", "Nd"),
                  regions=("omp", "outer_lower_target", "field_line"),
                  dpi=110, **kwargs):
    """Radial + parallel profile comparison via hermes3.plotting.lineplot.

    Each case's LAST-timestep profiles (lineplot selects t=-1 itself) overlaid
    in its own colour, across regions x params on ONE page (rows = regions,
    cols = params). This is the Hermes-3-only profile view -- no SOLPS/SOLEDGE
    reference line and no dependency on that import stack.

    Extra kwargs go straight to lineplot (ylims, lw, logscale, ...).

    lineplot reassigns its own `cases` dict to the t=-1 slice, so the fresh
    dict built here keeps the cached datasets untouched.
    """
    from hermes3.plotting import lineplot

    lineplot(
        ctx.cases.datasets(),
        colors=ctx.cases.color_list(),   # positional list, not a dict
        params=list(params),
        regions=list(regions),
        dpi=dpi,
        combine_regions=True,
        **kwargs,
    )
