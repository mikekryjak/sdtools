"""Profiles with a SOLPS reference line, via code_comparison.lineplot_compare.

Separate from the plain `profiles` page because it costs a lot more to import:
code_comparison hard-imports the SOLEDGE wrapper, which pulls in tkinter and is
absent from the hermes3 spack env. A campaign that lists this page must mock
tkinter before importing (see the flux_limiters campaign script); one that
doesn't list it pays nothing, which is the whole reason page imports are lazy.
"""

from . import register_page

# SOLPS references are expensive to read and usually shared across a campaign's
# studies, so cache by path for the life of the process.
_SOLPS_CACHE = {}


def load_solps(path):
    """Read (and cache) a SOLPS case as a code_comparison SOLPSdata."""
    if path not in _SOLPS_CACHE:
        from code_comparison.code_comparison import SOLPSdata
        sp = SOLPSdata()
        sp.read_from_case(path)
        _SOLPS_CACHE[path] = sp
    return _SOLPS_CACHE[path]


@register_page("profiles_vs_solps")
def profiles_vs_solps_page(ctx, reference=None, reference_label="SOLPS",
                           reference_color="black",
                           params=("Ne", "Te", "Na", "Ta", "NVd+"),
                           regions=("omp", "outer_fieldline_0.001_parallel",
                                    "outer_lower"),
                           dpi=100, **kwargs):
    """Last-timestep 1D profiles, cases overlaid, against a SOLPS reference.

    reference : path to the SOLPS case, or None for a Hermes-only page. None is
        the right setting for a study on a different machine than the reference
        was run for -- the comparison would be meaningless, and the page still
        renders with identical styling. Set it per study with
        @camp.study(page_opts={"profiles_vs_solps": dict(reference=None)}).

    Extra kwargs pass through to lineplot_compare (ylims, lw, legend_nrows,
    combine_molecules, ...).
    """
    from code_comparison.code_comparison import lineplot_compare

    cases = {n: dict(label=ctx.cases.label(n), color=ctx.cases.color(n))
             for n in ctx.cases.names}
    if reference is not None:
        cases = {reference_label: dict(data=load_solps(reference),
                                       color=reference_color),
                 **cases}

    lineplot_compare(
        cases=cases,
        data_dicts={"Hermes-3": ctx.cases.hermesdata()},
        regions=list(regions),
        params=list(params),
        dpi=dpi,
        **kwargs,
    )
