"""Cover page: what was compared, what was concluded, and what actually differs.

Version drift across a campaign is the reason this exists. The cover records
BOTH ends of it: the code that BUILT the PDF (analysis-time git SHAs) and the
code that RAN each case (parsed from BOUT.log.0), plus an automatic diff of
every option that differs across the cases -- which kills the "the case-name
token is just shorthand for what I changed" trap.
"""

import textwrap
from inspect import cleandoc

import matplotlib.pyplot as plt

from . import register_page
from .. import provenance
from ..report import raw_page, text_table, timestamp, priority_note


def _cover_text(ctx):
    cases = ctx.cases
    camp = ctx.campaign

    L = ["ANALYSIS ENVIRONMENT (at PDF build time)"]
    width = max((len(k) for k in camp.repos), default=0)
    for label, path in camp.repos.items():
        L.append(f"  {label.ljust(width)} : {provenance.git_describe(path)}")
    L.append("")

    L.append("CONCLUSIONS")
    for para in cleandoc(ctx.notes or "(none recorded)").splitlines():
        L += ["  " + wl for wl in (textwrap.wrap(para, 96) or [""])]
    L.append("")

    dirs = cases.dirs()

    L.append("CASES (as run)")
    crows = []
    for name in cases.names:
        ri = provenance.run_info(dirs[name])
        crows.append([
            cases.label(name), name, ri["date"], ri["hermes"], ri["bout"],
            f"CHK{provenance.check_level(dirs[name])}",
            provenance.run_status(dirs[name]),
        ])
    L += text_table(
        ["label", "sim id", "run date", "hermes", "BOUT++", "check", "status"],
        crows,
    ).splitlines()
    L.append("")

    diff, per, unrecorded, has_log = provenance.param_diff(
        dirs, priority=camp.param_diff_priority
    )
    nolog = [cases.label(n) for n in cases.names if not has_log[n]]
    if nolog:
        L.append("!! NO RUN LOG -- comparing BOUT.inp only for: "
                 + ", ".join(nolog))
        L += ["   " + wl for wl in textwrap.wrap(
            "BOUT.inp carries no code defaults, so a default-level difference "
            "against a case that DID run is not visible for these. Normal for "
            "a case that was prepared but never launched.", 93)]
        L.append("")

    order = priority_note(camp.param_diff_priority)
    L.append(f"DIFFERING OPTIONS  (what each run actually read: BOUT.log.0 over "
             f"BOUT.inp; all sections; run-provenance excluded; {order})")
    if diff:
        def clip(v):
            return v if len(v) <= 48 else v[:47] + "…"
        labels = [cases.label(n) for n in cases.names]
        drows = [[k, *(clip(per[n][k]) for n in cases.names)] for k in diff]
        L += text_table(["option", *labels], drows).splitlines()
    else:
        L.append("  (every option identical across cases, provenance aside)")

    if unrecorded:
        L.append("")
        L += ["   " + wl for wl in textwrap.wrap(
            "NOT COMPARABLE (" + str(len(unrecorded)) + "): values the code "
            "forces rather than reads are recorded only in a FINALISED "
            "BOUT.settings, so they are unavailable for a case that has not "
            "finished -- " + ", ".join(unrecorded), 93)]
    return "\n".join(L)


@register_page("cover")
def cover_page(ctx, fontsize=7):
    """Provenance cover: environment, conclusions, cases, differing options."""
    fig = plt.figure(figsize=ctx.page_size)
    fig.text(0.06, 0.96, ctx.slug, ha="left", va="top", fontsize=15,
             fontweight="bold")
    fig.text(0.06, 0.93,
             f"campaign: {ctx.campaign.name}    built {timestamp()}",
             ha="left", va="top", fontsize=8, color="0.35")
    fig.text(0.06, 0.89, _cover_text(ctx), ha="left", va="top",
             family="monospace", fontsize=fontsize)
    return raw_page(fig)
