"""Cover page: what was compared, what was concluded, and what actually differs.

Version drift across a campaign is the reason this exists. The cover records
BOTH ends of it: the code that BUILT the PDF (analysis-time git SHAs) and the
code that RAN each case (parsed from BOUT.log.0), plus an automatic diff of
every option that differs across the cases -- which kills the "the case-name
token is just shorthand for what I changed" trap.

Two styles, because a cover has two readers.

    style="full"   (default) every section above, as a monospace block, over as
                   many pages as it takes. What the diff table costs in pages,
                   it earns back the first time a study turns out to have
                   compared something other than what its case names claim.

    style="brief"  ONE page for a reader who has not run these cases: title,
                   date, what the report is, the conclusion, the configuration
                   and the case list, set in ordinary type. A run field that is
                   the same in every case (code version, CHECK level) is stated
                   once in the configuration block instead of repeating down a
                   column -- that collapse is most of what makes one page
                   possible. The option diff is not on it.

The detail the brief style leaves out is not lost: it is the "cover_details"
page, which a campaign lists LAST to keep the provenance at the back of the
report. Both styles and the appendix are built from the same section
functions, so the detail cannot drift from the cover that summarises it.
"""

import re
import textwrap
from inspect import cleandoc

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# Same arithmetic for the brief cover's proportional text: 0.5 em is the
# average advance of a proportional face, against 0.6 for monospace.
_PROP_CHAR_EM = 0.5

# Type sizes for the brief cover, in points.
_BRIEF_TITLE = 26
_BRIEF_SUBTITLE = 11
_BRIEF_HEADING = 12
_BRIEF_BODY = 11.5
_BRIEF_TABLE = 10
_BRIEF_TABLE_MIN = 6.5

# Longest line of body text, in characters. The report page is 14 inches wide,
# so a paragraph set to the full measure runs about 180 characters and is hard
# to track from one line to the next. This is a readability cap, not a margin.
_BRIEF_MEASURE = 100

# The brief cover's case-list colour chips, in figure fractions: the width of
# one chip, and how far the table text is indented to clear it.
_SWATCH_W = 0.010
_SWATCH_GAP = 0.018


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


# =============================================================================
# Sections -- each returns text lines, and each is used by more than one page
# =============================================================================
# Per-case run provenance, named once here so the full cover, the brief cover
# and the appendix cannot disagree about what a case ran.
_RUN_FIELDS = ("hermes", "bout", "check")
_FIELD_HEADER = {"hermes": "hermes", "bout": "BOUT++", "check": "CHECK"}


def _resolve_dirs(ctx, dirs):
    """Where every provenance read looks, for this study.

    The `dirs` hook exists for a campaign whose evidence is not the case
    directory: it redirects each read (BOUT.log.0, BOUT.settings, BOUT.inp) at
    whatever holds those files -- a results bundle, say. A campaign that keeps
    its runs after the dumps are deleted needs it.
    """
    return dirs(ctx) if callable(dirs) else ctx.cases.dirs()


def _run_fields(ctx, dirs):
    """{case name: {field: value}} -- run date, code versions, CHECK, status."""
    out = {}
    for name in ctx.cases.names:
        ri = provenance.run_info(dirs[name])
        out[name] = dict(
            date=ri["date"], hermes=ri["hermes"], bout=ri["bout"],
            check=provenance.check_level(dirs[name]),
            status=provenance.run_status(dirs[name]),
        )
    return out


def _environment_lines(camp):
    L = ["ANALYSIS ENVIRONMENT (at PDF build time)"]
    width = max((len(k) for k in camp.repos), default=0)
    for label, path in camp.repos.items():
        L.append(f"  {label.ljust(width)} : {provenance.git_describe(path)}")
    L.append("")
    return L


def _about_lines(ctx):
    L = ["ABOUT THIS REPORT"]
    if ctx.about:
        for para in cleandoc(ctx.about).splitlines():
            L += ["  " + wl for wl in (textwrap.wrap(para, 96) or [""])]
    else:
        L.append("  !! NO DESCRIPTION -- pass about= to @camp.study")
    L.append("")
    return L


def _conclusions_lines(ctx):
    L = ["CONCLUSIONS"]
    for para in cleandoc(ctx.notes or "(none recorded)").splitlines():
        L += ["  " + wl for wl in (textwrap.wrap(para, 96) or [""])]
    L.append("")
    return L


def _cases_lines(ctx, dirs, case_table):
    """The CASES table: one row per case, every run field as its own column.

    `case_table` replaces it with one built from a record, for a campaign that
    keeps its runs somewhere other than the case directory.
    """
    cases = ctx.cases
    L = ["CASES (as run)"]
    if callable(case_table):
        header, crows = case_table(ctx)
    else:
        header = ["label", "sim id", "run date", "hermes", "BOUT++", "check",
                  "status"]
        fields = _run_fields(ctx, dirs)
        crows = [[cases.label(n), n, fields[n]["date"], fields[n]["hermes"],
                  fields[n]["bout"], f"CHK{fields[n]['check']}",
                  fields[n]["status"]]
                 for n in cases.names]
    L += text_table(header, crows).splitlines()
    L.append("")
    return L


def _options_lines(ctx, dirs, max_chars, base):
    """The DIFFERING OPTIONS table and its caveats.

    `base` is the index this block starts at on the page, so the returned
    header map is in page coordinates and a page break can repeat the right
    table header.
    """
    cases = ctx.cases
    camp = ctx.campaign
    headers = {}
    L = []

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
        start = base + len(L)
        table = _wide_table(["option", *labels], drows, max_chars)
        L += [line for line, _ in table]
        headers.update({start + j: head for j, (_, head) in enumerate(table)
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


def _cover_text(ctx, case_table=None, dirs=None, max_chars=200):
    """(lines, headers) for the full cover -- headers maps a line index to the
    table header that must be repeated if a page break lands on it."""

    dirs = _resolve_dirs(ctx, dirs)
    L = _environment_lines(ctx.campaign)
    L += _about_lines(ctx)
    L += _conclusions_lines(ctx)
    L += _cases_lines(ctx, dirs, case_table)
    tail, headers = _options_lines(ctx, dirs, max_chars, base=len(L))
    return L + tail, headers


# =============================================================================
# Monospace page rendering (full cover and appendix)
# =============================================================================
def _mono_pages(ctx, lines, headers, fontsize, max_lines, head_first,
                head_cont):
    """Lay `lines` out as monospace pages, breaking without losing rows.

    Continuation pages rather than a block that runs off the bottom. A study
    with many cases overflows on both axes at once, and a page that silently
    loses its last rows is worse than one that takes two pages: the rows it
    drops are the differing options, which is the part nobody can reconstruct
    by eye.
    """
    figs, start = [], 0
    while start < max(len(lines), 1):
        # A break inside a table carries that table's header onto the next page,
        # so a continuation is still readable as a table.
        repeat = list(headers.get(start, ())) if start else []
        chunk = repeat + lines[start:start + max_lines - len(repeat)]
        start += max_lines - len(repeat)

        fig = plt.figure(figsize=ctx.page_size)
        head = head_first if not figs else head_cont
        fig.text(_LEFT, 0.96, head, ha="left", va="top", fontsize=15,
                 fontweight="bold")
        fig.text(_LEFT, 0.93,
                 f"campaign: {ctx.campaign.name}    built {timestamp()}",
                 ha="left", va="top", fontsize=8, color="0.35")
        fig.text(_LEFT, _TOP, "\n".join(chunk), ha="left", va="top",
                 family="monospace", fontsize=fontsize)
        figs.append(raw_page(fig))
    return figs


# =============================================================================
# Brief cover
# =============================================================================
def _frac(points, page_h):
    """`points` as a fraction of page height -- the unit every offset is in."""
    return points / (page_h * 72.0)


def _prop_chars(page_w, fontsize, measure=_BRIEF_MEASURE):
    """How many characters of proportional text to set on one line."""
    width_pt = (1.0 - _LEFT - _MARGIN) * page_w * 72
    fits = max(int(width_pt / (fontsize * _PROP_CHAR_EM)), 20)
    return min(fits, measure)


def _split_slug(slug):
    """("05", "Test4 mc vs va") from "05_test4_mc_vs_va".

    The study function name is all the title there is, so it is spelled out
    rather than dressed up: underscores become spaces, the first letter is
    capitalised, nothing else.
    """
    m = re.match(r"^(\d+)[_-](.*)$", slug)
    num, name = (m.group(1), m.group(2)) if m else ("", slug)
    title = name.replace("_", " ").strip() or slug
    return num, title[:1].upper() + title[1:]


def _draw(fig, y, page_h, lines, fontsize, *, x=_LEFT, color="black",
          weight="normal", line_em=1.35):
    """Draw a block of text at `y` and return the y below it."""
    if not lines:
        return y
    fig.text(x, y, "\n".join(lines), ha="left", va="top", fontsize=fontsize,
             color=color, fontweight=weight, linespacing=line_em)
    return y - len(lines) * _frac(fontsize * line_em, page_h)


def _wrap(text, page_w, fontsize):
    """Paragraph text as lines, keeping the author's own line breaks."""
    out = []
    for para in cleandoc(text).splitlines():
        out += textwrap.wrap(para, _prop_chars(page_w, fontsize)) or [""]
    return out


def _brief_case_table(ctx, dirs, case_table):
    """(header, rows, names, common) for the brief cover's case list.

    A run field with the same value in every case is NOT given a column: it
    moves into `common`, which the configuration block states once. Only a
    field that actually differs earns a column, which is what lets the list fit
    on one page. `names` is the case name per row, for the colour chips, or
    None when the campaign supplied its own table and the rows cannot be
    mapped back to cases.
    """
    cases = ctx.cases
    if callable(case_table):
        header, rows = case_table(ctx)
        return header, rows, None, {}

    fields = _run_fields(ctx, dirs)
    names = cases.names
    common, varying = {}, []
    for key in _RUN_FIELDS:
        values = {fields[n][key] for n in names}
        if len(values) == 1:
            common[key] = values.pop()
        else:
            varying.append(key)

    header = ["case", "simulation", "run date",
              *(_FIELD_HEADER[k] for k in varying), "status"]
    rows = [[cases.label(n), n, fields[n]["date"],
             *(fields[n][k] for k in varying), fields[n]["status"]]
            for n in names]
    return header, rows, names, common


def _config_lines(camp, common):
    """The configuration block: what every case shares, and what built the PDF."""
    lines = []
    if common:
        parts = []
        if common.get("hermes", "?") != "?":
            parts.append(f"Hermes-3 {common['hermes']}")
        if common.get("bout", "?") != "?":
            parts.append(f"BOUT++ {common['bout']}")
        if common.get("check", "?") != "?":
            parts.append(f"CHECK={common['check']}")
        if parts:
            lines.append("every case ran with " + ", ".join(parts))
    if camp.repos:
        lines.append("report built with " + ", ".join(
            f"{label} {provenance.git_describe(path)}"
            for label, path in camp.repos.items()))
    return lines


def _table_chars(page_w, fontsize):
    """How many monospace characters fit on the brief cover's table line."""
    width_pt = (1.0 - _LEFT - _SWATCH_GAP - _MARGIN) * page_w * 72
    return int(width_pt / (fontsize * _CHAR_EM))


def _block_width(block):
    return max((len(line) for line in block), default=0)


def _clip_columns(header, rows, max_chars):
    """Shorten the widest columns until the table fits `max_chars`.

    Returns (rows, clipped). A shortened value ends in "…", the same mark the
    option table uses, so a truncated name never reads as a real one. The case
    name (first column) is shortened only after everything else has been: it is
    what the reader matches against the plots. Nothing is dropped -- the
    appendix page carries every value in full.
    """
    cells = [header, *rows]
    widths = [max(len(str(r[i])) for r in cells) for i in range(len(header))]

    def total():
        return sum(widths) + 2 * (len(widths) - 1)

    def floor(i):
        return max(len(str(header[i])), 12)

    # Always take the character from whichever column is widest now, so a long
    # column is brought down towards the others rather than one column being
    # cut to the bone while its neighbour keeps every character.
    clipped = False
    while total() > max_chars:
        movable = [i for i in range(1, len(header)) if widths[i] > floor(i)]
        if not movable and widths[0] > floor(0):
            movable = [0]
        if not movable:
            break
        i = max(movable, key=lambda c: widths[c])
        widths[i] -= 1
        clipped = True
    if not clipped:
        return rows, False
    return [[v if len(str(v)) <= w else str(v)[:w - 1] + "…"
             for v, w in zip(map(str, row), widths)] for row in rows], True


def _chunk_table(block, per_page):
    """Split a text table into page-sized chunks, repeating its header.

    Returns [(lines, block indices)], where a repeated header line has index
    None. The indices are what maps a drawn line back to its case, for the
    colour chips.
    """
    chunks, i = [], 0
    while i < len(block):
        repeat = [] if not chunks else block[:2]
        take = max(per_page - len(repeat), 1)
        lines = repeat + block[i:i + take]
        index = [None] * len(repeat) + list(range(i, min(i + take, len(block))))
        chunks.append((lines, index))
        i += take
    return chunks


def _draw_rows(fig, top, page_h, lines, fontsize, x):
    """Draw a monospace table one line at a time, and return the y below it.

    Line by line rather than as one text block because matplotlib spaces the
    lines of a block by the FONT's line height, not by the point size, and the
    colour chips are placed from the point size. Over a couple of dozen cases
    the two models drift a full row apart, which puts every chip against the
    wrong case. Placing each line puts both on the same arithmetic.
    """
    advance = _frac(fontsize * _LINE_EM, page_h)
    for i, line in enumerate(lines):
        fig.text(x, top - i * advance, line, ha="left", va="top",
                 family="monospace", fontsize=fontsize)
    return top - len(lines) * advance


def _draw_chips(fig, top, fontsize, page_h, colors):
    """A colour chip per case row, so the list keys the plots that follow."""
    advance = _frac(fontsize * _LINE_EM, page_h)
    height = advance * 0.55
    for row, color in colors.items():
        centre = top - (row + 0.5) * advance
        fig.add_artist(Rectangle((_LEFT, centre - height / 2), _SWATCH_W,
                                 height, transform=fig.transFigure,
                                 facecolor=color, edgecolor="none",
                                 clip_on=False))


def _brief_pages(ctx, case_table=None, dirs=None):
    page_w, page_h = ctx.page_size
    cases = ctx.cases
    camp = ctx.campaign
    dirs = _resolve_dirs(ctx, dirs)
    header, rows, names, common = _brief_case_table(ctx, dirs, case_table)

    num, title = _split_slug(ctx.slug)
    subtitle = f"campaign {camp.name}"
    if num:
        subtitle += f"  ·  report {num}"
    subtitle += f"  ·  built {timestamp()}"

    fig = plt.figure(figsize=ctx.page_size)
    y = 0.93
    y = _draw(fig, y, page_h, [title], _BRIEF_TITLE, weight="bold",
              line_em=1.25)
    y = _draw(fig, y, page_h, [subtitle], _BRIEF_SUBTITLE, color="0.35")
    y -= _frac(6, page_h)
    fig.add_artist(plt.Line2D([_LEFT, 1 - _MARGIN], [y, y], color="0.75",
                              lw=0.8, transform=fig.transFigure))

    def section(y, heading, lines, color="black"):
        y -= _frac(_BRIEF_HEADING * 1.6, page_h)
        y = _draw(fig, y, page_h, [heading.upper()], _BRIEF_HEADING,
                  weight="bold", color="0.35")
        y -= _frac(2, page_h)
        return _draw(fig, y, page_h, lines, _BRIEF_BODY, x=_LEFT + 0.004,
                     color=color)

    if ctx.about:
        y = section(y, "About this report", _wrap(ctx.about, page_w,
                                                 _BRIEF_BODY))
    else:
        y = section(y, "About this report",
                    ["(no description -- pass about= to @camp.study)"],
                    color="0.45")
    y = section(y, "Conclusions",
                _wrap(ctx.notes or "(none recorded)", page_w, _BRIEF_BODY))

    config = _config_lines(camp, common)
    if config:
        y = section(y, "Configuration", config)

    # The case list is the one block that can outgrow the page. Shrink it
    # first; if it still does not fit, continue it on a second page. Dropping
    # rows is not an option -- a cover that quietly omits a case is exactly the
    # failure the full cover was built to prevent.
    y -= _frac(_BRIEF_HEADING * 1.6, page_h)
    y = _draw(fig, y, page_h, ["CASES"], _BRIEF_HEADING, weight="bold",
              color="0.35")
    y -= _frac(4, page_h)

    block = text_table(header, rows).splitlines()
    fontsize = _BRIEF_TABLE
    available = y - _MARGIN
    while fontsize > _BRIEF_TABLE_MIN and (
            len(block) * _frac(fontsize * _LINE_EM, page_h) > available
            or _block_width(block) > _table_chars(page_w, fontsize)):
        fontsize -= 0.5
    rows, clipped = _clip_columns(header, rows, _table_chars(page_w, fontsize))
    if clipped:
        block = text_table(header, rows).splitlines()
    per_page = max(int(available / _frac(fontsize * _LINE_EM, page_h)), 3)

    figs = []
    chunks = _chunk_table(block, per_page)
    for chunk, (lines, index) in enumerate(chunks):
        if chunk:
            fig = plt.figure(figsize=ctx.page_size)
            y = 0.93
            y = _draw(fig, y, page_h, [f"{title} -- cases continued"],
                      _BRIEF_HEADING + 3, weight="bold")
            y -= _frac(8, page_h)
        top = y
        end = _draw_rows(fig, top, page_h, lines, fontsize,
                         _LEFT + _SWATCH_GAP)
        if names is not None:
            _draw_chips(fig, top, fontsize, page_h,
                        {j: cases.color(names[b - 2])
                         for j, b in enumerate(index)
                         if b is not None and b >= 2})
        if clipped and chunk == len(chunks) - 1:
            _draw(fig, end - _frac(4, page_h), page_h,
                  ["… = shortened to fit; the provenance appendix lists every "
                   "value in full"], _BRIEF_SUBTITLE - 1,
                  x=_LEFT + _SWATCH_GAP, color="0.45")
        figs.append(raw_page(fig))
    return figs


# =============================================================================
# Pages
# =============================================================================
@register_page("cover")
def cover_page(ctx, fontsize=7, case_table=None, dirs=None, style="full"):
    """Cover: environment, about, conclusions, cases, differing options.

    style : "full" (default) or "brief"
        "full" prints every section as a monospace block, over as many pages as
        it takes. "brief" prints one page for a reader who has not run these
        cases -- title, date, about, conclusions, configuration and case list --
        and leaves the option diff to the "cover_details" page. See the module
        docstring.
    fontsize : point size of the monospace block, "full" only.
    case_table : callable(ctx) -> (header, rows), optional
        Replaces the CASES table, for a campaign that records its runs
        somewhere other than the case directory. The brief cover shows such a
        table as given, without colour chips or the common-field collapse.
    dirs : callable(ctx) -> {case name: directory}, optional
        Where to read each case's BOUT.log.0 / BOUT.settings / BOUT.inp from.
        Defaults to the case directories themselves.
    """
    if style not in ("full", "brief"):
        raise ValueError(f"cover style must be 'full' or 'brief', got {style!r}")
    if style == "brief":
        return _brief_pages(ctx, case_table, dirs)

    max_chars, max_lines = _page_capacity(ctx.page_size, fontsize)
    lines, headers = _cover_text(ctx, case_table, dirs, max_chars)
    return _mono_pages(ctx, lines, headers, fontsize, max_lines, ctx.slug,
                       f"{ctx.slug}  (cover continued)")


@register_page("cover_details")
def cover_details_page(ctx, fontsize=7, case_table=None, dirs=None):
    """Provenance appendix: environment, cases as run, differing options.

    What the brief cover leaves out, for a campaign that wants the detail kept
    but at the BACK of the report rather than in front of it. List it last in
    the campaign's pages, beside a cover with style="brief". A full cover
    already carries this content, so listing both repeats it.

    Takes the same fontsize / case_table / dirs options as the cover.
    """
    max_chars, max_lines = _page_capacity(ctx.page_size, fontsize)
    dirs = _resolve_dirs(ctx, dirs)
    lines = _environment_lines(ctx.campaign)
    lines += _cases_lines(ctx, dirs, case_table)
    tail, headers = _options_lines(ctx, dirs, max_chars, base=len(lines))
    return _mono_pages(ctx, lines + tail, headers, fontsize, max_lines,
                       f"{ctx.slug}  (provenance appendix)",
                       f"{ctx.slug}  (provenance appendix continued)")
