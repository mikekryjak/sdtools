"""Solver performance page: speed histories + ddt() residual histories."""

from . import register_page
from ..report import stack_figures


@register_page("performance")
def performance_page(ctx, ddt=(), stack=True, logscale=True):
    """plot_performance (sim-time vs wall-time, speed in ms/24hr) plus, if
    `ddt` names any residuals, compare_ddt (RMS ddt residual histories).

    ddt : names WITHOUT the ddt() wrapper, e.g. ("Pe", "Pd", "Nd+"). Filtered
        to the diagnostics EVERY case actually saved -- ddt() fields need
        diagnose=true under [solver] at run time, so a solver recipe without
        diagnostics just drops the panel instead of failing the page.
    stack : True puts both plots on one PDF page; False gives them a page each.
    """
    import matplotlib.pyplot as plt
    from hermes3.plotting import plot_performance, compare_ddt

    datasets = ctx.cases.datasets()
    colors = ctx.cases.colors()

    # Both plotters create figures and return nothing, so track them by number.
    before = set(plt.get_fignums())

    plot_performance(datasets, colors=colors, logscale=logscale)

    available = [p for p in ddt if ctx.cases.has_var(f"ddt({p})")]
    skipped = [p for p in ddt if p not in available]
    if skipped:
        print(f"  [page:performance] ddt() not saved by every case, "
              f"skipped: {skipped}")
    if available:
        compare_ddt(datasets, available, colors=colors)

    figs = [plt.figure(n) for n in plt.get_fignums() if n not in before]
    if stack and len(figs) > 1:
        return stack_figures(figs)
    return figs
