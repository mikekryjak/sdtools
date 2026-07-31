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


# Monospace advance is 0.6 em and line spacing 1.2 em for every face matplotlib
# is likely to pick. Arithmetic rather than measurement, deliberately: measuring
# needs a renderer, and the text must be laid out before anything is drawn.
_CHAR_EM = 0.6
_LINE_EM = 1.2
_LEFT = 0.06
_TOP = 0.89
_MARGIN = 0.04


def _page_capacity(page_size, fontsize):
    """(characters per line, lines per page) for the cover's monospace block."""

    width_pt = (1.0 - _LEFT - _MARGIN) * page_size[0] * 72
    height_pt = (_TOP - _MARGIN) * page_size[1] * 72
    return (int(width_pt / (fontsize * _CHAR_EM)),
            int(height_pt / (fontsize * _LINE_EM)))


def _wide_table(headers, rows, max_chars):
    """`text_table`, split into column groups so that no line runs off the page.

    The first column repeats in every group, because it names the row and a
    group without it cannot be read. A single column too wide even on its own is
    emitted regardless: the values are already clipped by the caller, and
    dropping one would lose a difference the page exists to show.
    """

    def build(cols):
        return text_table([headers[0], *(headers[c] for c in cols)],
                          [[r[0], *(r[c] for c in cols)] for r in rows])

    def width(block):
        return max((len(line) for line in block.splitlines()), default=0)

    groups, col = [], 1
    while col < len(headers):
        take = [col]
        while col + len(take) < len(headers):
            if width(build(take + [col + len(take)])) > max_chars:
                break
            take.append(col + len(take))
        groups.append(take)
        col += len(take)

    # Each line is paired with its group's header, so that a vertical page break
    # can repeat it. A continuation page of bare values with no column names is
    # not a table, it is a wall of numbers.
    out = []
    for i, cols in enumerate(groups):
        if i:
            out += [("", None),
                    (f"  ...DIFFERING OPTIONS continued, cases "
                     f"{cols[0]}-{cols[-1]} of {len(headers) - 1}", None)]
        block = build(cols).splitlines()
        head = tuple(block[:2])  # header row + rule
        out += [(line, head) for line in block]
    return out


def _cover_text(ctx, case_table=None, dirs=None, max_chars=200):
    """(lines, headers) -- headers maps a line index to the table header that
    must be repeated if a page break lands on it."""

    cases = ctx.cases
    camp = ctx.campaign
    headers = {}

    L = ["ANALYSIS ENVIRONMENT (at PDF build time)"]
    width = max((len(k) for k in camp.repos), default=0)
    for label, path in camp.repos.items():
        L.append(f"  {label.ljust(width)} : {provenance.git_describe(path)}")
    L.append("")

    L.append("CONCLUSIONS")
    for para in cleandoc(ctx.notes or "(none recorded)").splitlines():
        L += ["  " + wl for wl in (textwrap.wrap(para, 96) or [""])]
    L.append("")

    # Both hooks exist for a campaign whose evidence is not the case directory.
    # `dirs` redirects every provenance read (BOUT.log.0, BOUT.settings,
    # BOUT.inp) at whatever holds those files -- a results bundle, say -- and
    # `case_table` replaces the CASES table with one built from a record. A
    # campaign that keeps its runs after the dumps are deleted needs both.
    dirs = dirs(ctx) if callable(dirs) else cases.dirs()

    L.append("CASES (as run)")
    if callable(case_table):
        header, crows = case_table(ctx)
    else:
        header = ["label", "sim id", "run date", "hermes", "BOUT++", "check",
                  "status"]
        crows = []
        for name in cases.names:
            ri = provenance.run_info(dirs[name])
            crows.append([
                cases.label(name), name, ri["date"], ri["hermes"], ri["bout"],
                f"CHK{provenance.check_level(dirs[name])}",
                provenance.run_status(dirs[name]),
            ])
    L += text_table(header, crows).splitlines()
    L.append("")

    diff, per, unrecorded, derived, has_log = provenance.param_diff(
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
        base = len(L)
        table = _wide_table(["option", *labels], drows, max_chars)
        L += [line for line, _ in table]
        headers.update({base + j: head for j, (_, head) in enumerate(table)
                        if head})
    else:
        L.append("  (every option identical across cases, provenance aside)")

    if derived:
        L.append("")
        L += ["   " + wl for wl in textwrap.wrap(
            "DERIVED, NOT CHOSEN (" + str(len(derived)) + "): nobody set "
            "these. They differ because BOUT computes their defaults from "
            "options that DO differ above, so each one restates a change "
            "already listed -- " + ", ".join(derived), 93)]

    if unrecorded:
        L.append("")
        L += ["   " + wl for wl in textwrap.wrap(
            "NOT COMPARABLE (" + str(len(unrecorded)) + "): values the code "
            "forces rather than reads are recorded only in a FINALISED "
            "BOUT.settings, so they are unavailable for a case that has not "
            "finished -- " + ", ".join(unrecorded), 93)]
    return L, headers


@register_page("cover")
def cover_page(ctx, fontsize=7, case_table=None, dirs=None):
    """Provenance cover: environment, conclusions, cases, differing options.

    case_table : callable(ctx) -> (header, rows), optional
        Replaces the CASES table, for a campaign that records its runs
        somewhere other than the case directory.
    dirs : callable(ctx) -> {case name: directory}, optional
        Where to read each case's BOUT.log.0 / BOUT.settings / BOUT.inp from.
        Defaults to the case directories themselves.
    """
    max_chars, max_lines = _page_capacity(ctx.page_size, fontsize)
    lines, headers = _cover_text(ctx, case_table, dirs, max_chars)

    # Continuation pages rather than a block that runs off the bottom. A study
    # with many cases overflows on both axes at once, and a cover that silently
    # loses its last rows is worse than a cover that takes two pages: the rows
    # it drops are the differing options, which is the part nobody can
    # reconstruct by eye.
    figs, start = [], 0
    while start < max(len(lines), 1):
        # A break inside a table carries that table's header onto the next page,
        # so a continuation is still readable as a table.
        repeat = list(headers.get(start, ())) if start else []
        chunk = repeat + lines[start:start + max_lines - len(repeat)]
        start += max_lines - len(repeat)

        fig = plt.figure(figsize=ctx.page_size)
        head = ctx.slug if not figs else f"{ctx.slug}  (cover continued)"
        fig.text(_LEFT, 0.96, head, ha="left", va="top", fontsize=15,
                 fontweight="bold")
        fig.text(_LEFT, 0.93,
                 f"campaign: {ctx.campaign.name}    built {timestamp()}",
                 ha="left", va="top", fontsize=8, color="0.35")
        fig.text(_LEFT, _TOP, "\n".join(chunk), ha="left", va="top",
                 family="monospace", fontsize=fontsize)
        figs.append(raw_page(fig))
    return figs
