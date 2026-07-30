"""PDF assembly: turn a list of pages into one self-contained study PDF.

Page contract
-------------
A page is a callable ``page(ctx, **opts)``. It may

  * return a matplotlib Figure, or a list of Figures (one PDF page each), or
  * return None and simply leave its figures open on the pyplot stack.

The second form exists so that EXISTING sdtools plotting functions work as
pages with a two-line adapter: most of them (lineplot, plot_performance,
compare_ddt, ...) create figures and return nothing. The runner snapshots the
open figure numbers before calling the page and treats anything new as that
page's output, so nothing has to be rewritten to be usable here.

Figures are rasterised and letterboxed onto a fixed page size, so paging
through a PDF doesn't make the page size jump around (source figures vary from
tall narrow field maps to wide multi-panel profile grids). A page that has
already laid itself out at the right size -- the cover -- opts out via
``raw_page(fig)``.
"""

import io
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# Figure helpers available to page authors
# =============================================================================
def raw_page(fig, rc=None):
    """Mark `fig` to be written to the PDF verbatim, skipping letterboxing.

    For figures already sized to the report page (the cover, the scan pages).
    Returns `fig` so it can be used inline: ``return raw_page(fig)``.

    rc : rcParams the figure was LAID OUT under, to be re-applied when it is
        saved. This matters more than it looks: tick locators re-evaluate at
        DRAW time, so a page that sized its text with an rc_context and then
        let the report save it outside that context gets a different number of
        ticks than it laid out for. Pass the same dict here and the two agree.
    """
    fig._cc_raw = True
    if rc:
        fig._cc_rc = dict(rc)
    return fig


def stack_figures(figs=None, pad=10, dpi=150):
    """Rasterise several figures and stack them vertically into ONE new figure.

    For pages built from more than one plotting call (e.g. plot_performance +
    compare_ddt) that should land on a single PDF page. The originals are
    closed; the new stacked figure is returned. `figs` defaults to every
    currently-open figure.
    """
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    figs = [f for f in figs if f is not None]
    if not figs:
        return None

    def collapse_white(img):
        """Trim horizontal bands of pure white, keeping `pad` rows of margin --
        otherwise stacked figures are mostly each other's whitespace."""
        row_has_ink = np.any(img[..., :3] < 250, axis=(1, 2))
        if not row_has_ink.any():
            return img
        keep = np.convolve(row_has_ink.astype(int), np.ones(2 * pad + 1), "same") > 0
        return img[keep]

    images = []
    for fig in figs:
        fig.set_dpi(dpi)
        fig.canvas.draw()
        images.append(collapse_white(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)

    width = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        gap = width - img.shape[1]
        if gap:
            img = np.pad(img, ((0, 0), (0, gap), (0, 0)), constant_values=255)
        padded.append(img)
    stacked = np.vstack(padded)

    h, w = stacked.shape[:2]
    out = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = out.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.imshow(stacked, aspect="auto")
    return out


def text_table(headers, rows):
    """Monospace-aligned text table (for text pages such as the cover)."""
    cols = list(zip(headers, *rows)) if rows else [[h] for h in headers]
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt(r):
        return "  ".join(str(c).ljust(w) for c, w in zip(r, widths))

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([fmt(headers), sep, *(fmt(r) for r in rows)])


# =============================================================================
# Page context
# =============================================================================
class PageContext:
    """What every page is handed: the study's data collection plus its identity.

    Attributes
    ----------
    data      : the collection -- a CaseSet for a case campaign, a SeriesSet
                for a scan campaign
    cases     : alias of `data`, read by case pages
    series    : alias of `data`, read by scan pages
    slug      : str -- "NN_study_name", also the PDF stem
    notes     : str -- the study's written conclusion (its docstring)
    campaign  : Campaign
    page_size : (w, h) inches

    The two aliases exist so a page reads as what it actually plots
    (`ctx.cases.datasets()` vs `ctx.series.labels`) while the campaign machinery
    handles one object.
    """

    def __init__(self, data, slug, notes, campaign):
        self.data = data
        self.slug = slug
        self.notes = notes
        self.campaign = campaign
        self.page_size = campaign.page_size

    @property
    def cases(self):
        return self.data

    @property
    def series(self):
        return self.data


# =============================================================================
# Report
# =============================================================================
class Report:
    """Builds one study PDF from a list of resolved (page callable, opts)."""

    def __init__(self, path, page_size=(14.0, 9.5), dpi=150):
        self.path = path
        self.page_size = page_size
        self.dpi = dpi

    def build(self, pages, ctx):
        self._page_no = 0
        with PdfPages(self.path) as pdf:
            for fn, opts in pages:
                name = getattr(fn, "__name__", str(fn))
                try:
                    figs = self._run_page(fn, ctx, opts)
                except Exception as e:  # noqa: BLE001
                    # One broken page must not cost the whole study PDF: report
                    # it, drop its figures, carry on with the rest.
                    print(f"  [page:{name}] FAILED: {type(e).__name__}: {e}")
                    plt.close("all")
                    continue
                for fig in figs:
                    self._write(pdf, fig)
        return self.path

    def _run_page(self, fn, ctx, opts):
        """Call one page and collect its figures (see the module docstring)."""
        before = set(plt.get_fignums())
        returned = fn(ctx, **opts)
        new = [plt.figure(n) for n in plt.get_fignums() if n not in before]

        if returned is None:
            figs = new
        elif isinstance(returned, plt.Figure):
            figs = [returned]
        else:
            figs = list(returned)

        # Close anything the page created but didn't hand back, so an orphan
        # can't leak into the next page's figure delta.
        keep = {id(f) for f in figs}
        for f in new:
            if id(f) not in keep:
                plt.close(f)
        return figs

    def _stamp(self, fig):
        """Number the page, counting the cover as 1 so the mark matches what a
        PDF viewer shows -- otherwise "page 5" means two different pages
        depending on who is looking."""
        self._page_no += 1
        fig.text(0.985, 0.012, str(self._page_no), ha="right", va="bottom",
                 fontsize=9, color="0.45")

    def _write(self, pdf, fig):
        # rcParams the page laid itself out under, re-applied for the draw that
        # savefig triggers -- otherwise tick locators, which run at draw time,
        # disagree with the layout (see raw_page).
        page_rc = getattr(fig, "_cc_rc", None) or {}

        if getattr(fig, "_cc_raw", False):
            self._stamp(fig)
            # sdtools' general/plotstyle.py sets rcParams["savefig.bbox"] =
            # "tight" globally, which would crop a full-page text figure back
            # to its text content unless overridden here.
            with plt.rc_context({**page_rc, "savefig.bbox": None}):
                pdf.savefig(fig)
            plt.close(fig)
            return

        buf = io.BytesIO()
        with plt.rc_context(page_rc):
            fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = plt.imread(buf)[..., :3]  # drop alpha

        img_h, img_w = img.shape[:2]
        img_aspect = img_w / img_h
        page_aspect = self.page_size[0] / self.page_size[1]
        if img_aspect > page_aspect:  # relatively wider -> width-limited
            disp_w, disp_h = 1.0, page_aspect / img_aspect
        else:  # relatively taller -> height-limited
            disp_w, disp_h = img_aspect / page_aspect, 1.0
        left, bottom = (1 - disp_w) / 2, (1 - disp_h) / 2

        page = plt.figure(figsize=self.page_size)
        ax = page.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.imshow(img, extent=(left, left + disp_w, bottom, bottom + disp_h),
                  aspect="auto")
        self._stamp(page)
        with plt.rc_context({"savefig.bbox": None}):
            pdf.savefig(page, dpi=self.dpi)
        plt.close(page)


def priority_note(priority, maxshow=4):
    """Short human description of the diff ordering, for a page header.

    A campaign's priority list can be long (a dozen legacy option names); it
    ran off the page when spelled out in full, so cap it.
    """
    if not priority:
        return "alphabetical"
    shown = ", ".join(priority[:maxshow])
    extra = f" (+{len(priority) - maxshow} more)" if len(priority) > maxshow else ""
    return f"{shown}{extra} first"


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
