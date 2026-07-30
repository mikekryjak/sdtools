"""Campaign -- one focus, one page set, many studies.

A *campaign* is a directory of related simulations analysed the same way. It
owns the page set (declared once, so every study shows the same plots and
studies stay comparable) and the case root the study names resolve against.

A *study* is a case selection plus a written conclusion, emitted as ONE
self-contained PDF with a provenance cover. It is declared once as an
``@camp.study`` function: the function name becomes the PDF name, its docstring
the cover notes, and its position in the file its number.

    from case_comparison import Campaign

    camp = Campaign(
        name="perftests",
        case_root="/home/mike/work/cases",
        pages=[
            "cover",
            ("profiles", dict(params=["Ne", "Te"], regions=["omp"])),
            ("performance", dict(ddt=["Pe", "Pd"])),
            "runtime_breakdown",
        ],
    )

    @camp.study
    def mc_vs_va():
        '''VA is faster after the slow initial phase.'''
        return {"case-dir-mc": "MC limiter", "case-dir-va": "VanAlbada"}

    if __name__ == "__main__":
        camp.cli()
"""

import sys
import inspect
import pathlib

from . import pages as page_registry
from .caseset import CaseLoader, CaseSet, DEFAULT_PALETTE
from .report import PageContext, Report


def _caller_dir(depth=2):
    """Directory of the script that constructed the Campaign, for OUTDIR."""
    try:
        frame = inspect.stack()[depth]
        f = frame.frame.f_globals.get("__file__")
        if f:
            return pathlib.Path(f).resolve().parent
    except Exception:
        pass
    return pathlib.Path.cwd()


def resolve_pages(pages, page_opts=None):
    """Normalise a campaign `pages` list to [(callable, opts dict)].

    Each entry may be:
        "name"                      built-in page, default options
        ("name", {...})             built-in page with options
        callable                    a campaign-local page function
        (callable, {...})           ditto with options

    Campaign-local callables are how anything study- or campaign-specific stays
    OUT of shared code: write the function in the campaign script and list it
    here. Nothing bespoke ever needs to be edited into case_comparison.

    page_opts : {page key: {...}}, optional
        Per-study option patches merged over the campaign defaults, keyed by
        page name (built-ins) or function __name__ (local callables). For the
        one study that needs a page configured differently -- see
        Campaign.study(page_opts=...).
    """
    page_opts = page_opts or {}
    out = []
    for entry in pages:
        opts = {}
        if isinstance(entry, tuple):
            if len(entry) != 2 or not isinstance(entry[1], dict):
                raise TypeError(
                    f"page entry {entry!r} must be (name_or_callable, opts dict)"
                )
            entry, opts = entry
        fn = page_registry.resolve(entry) if isinstance(entry, str) else entry
        if not callable(fn):
            raise TypeError(f"page entry {entry!r} is neither a name nor callable")
        key = entry if isinstance(entry, str) else getattr(fn, "__name__", None)
        merged = dict(opts)
        merged.update(page_opts.get(key, {}))
        out.append((fn, merged))

    keys = set()
    for e in pages:
        if isinstance(e, tuple):
            e = e[0]
        keys.add(e if isinstance(e, str) else getattr(e, "__name__", None))
    unknown = set(page_opts) - keys
    if unknown:
        raise KeyError(
            f"page_opts names no page in this campaign's page set: "
            f"{sorted(unknown)}"
        )
    return out


class Campaign:
    """A campaign: its case root, its page set, and its registered studies.

    Parameters
    ----------
    name : str
        Campaign name, printed on every cover.
    case_root : str
        Root searched recursively for simulation directories (a case is keyed
        by its directory NAME, so intermediate levels don't appear in study
        definitions).
    pages : list
        The shared page set -- see resolve_pages for the accepted forms.
        Edit here to change every study at once.
    grid_root : str, optional
        Root for grid files; defaults to case_root.
    outdir : path, optional
        Where PDFs are written; defaults to <campaign script dir>/output.
    repos : dict, optional
        {label: repo path} whose git SHAs are stamped on each cover, recording
        which analysis code built the PDF.
    param_diff_priority : sequence of str, optional
        DISPLAY-ORDER HINT ONLY for the cover's differing-options table, never
        a filter (every option of every case is always compared). Entries match
        either as a flattened-key prefix ("solver:", "petsc:") or as a bare
        option name ("flux_limit" matches "d:flux_limit").
    palette : list, optional
        Line colours auto-assigned to cases after the first (which is black).
    page_size : (w, h) inches
        Every PDF page is letterboxed to this, so the page size doesn't jump
        around while browsing.
    load_kwargs : dict, optional
        Extra kwargs passed to CaseDB.load_case_2D.
    """

    DEFAULT_REPOS = {
        "hermes-3": "/home/mike/work/hermes-3",
        "sdtools": "/home/mike/work/sdtools",
    }

    def __init__(self, name, case_root, pages, grid_root=None, outdir=None,
                 repos=None, param_diff_priority=(), palette=None,
                 page_size=(14.0, 9.5), load_kwargs=None,
                 collection=None, collection_opts=None, numbered=True,
                 needs_dumps=True, dir_fallback=None):
        self.name = name
        self.pages = list(pages)
        # needs_dumps=False for a campaign whose pages read no simulation
        # output at all -- every number comes from logs or an archived record.
        # It skips the pre-load below, which would otherwise open every case's
        # dataset for nothing and fail outright once the output is deleted.
        # Pages still get datasets on demand, so this is a pre-load switch, not
        # a ban. Default True: a campaign that plots fields must keep failing
        # early rather than half-way through a report.
        self.needs_dumps = needs_dumps
        # numbered=True gives PDFs an "NN_" prefix recording the study's
        # position in the file (an append-only chronological record). Set False
        # for a campaign whose PDFs are named by subject alone. Studies still
        # carry numbers either way, so the CLI can select by index.
        self.numbered = numbered
        # What a study's returned dict is turned into. CaseSet (the default)
        # compares named cases in detail; SeriesSet compares whole scans as
        # lines. collection_opts carries whatever that type needs beyond the
        # common (spec, loader, palette) -- e.g. a SeriesSet's sweep-coordinate
        # pattern and scalar function.
        self.collection = collection or CaseSet
        self.collection_opts = dict(collection_opts or {})
        self.repos = dict(self.DEFAULT_REPOS if repos is None else repos)
        self.param_diff_priority = tuple(param_diff_priority)
        self.palette = list(palette or DEFAULT_PALETTE)
        self.page_size = page_size

        self.outdir = pathlib.Path(outdir) if outdir else _caller_dir() / "output"
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.loader = CaseLoader(case_root, grid_root, load_kwargs,
                                 dir_fallback=dir_fallback)
        self._studies = {}  # function name -> dict(fn=, num=)

    # --- study registration -------------------------------------------------
    def study(self, fn=None, *, num=None, name=None, pages=None, page_opts=None):
        """Register a study. The function returns its cases dict
        ({sim dir name: label-or-dict}); everything else is derived:

          * PDF name    = NN_<function name>  (output/NN_<function name>.pdf)
          * cover notes = the function's docstring
          * NN          = registration order, i.e. position in the file -- an
                          append-only chronological record, so numbers stay
                          stable as long as new studies go at the bottom.

        Overrides, for studies whose identity is already fixed by PDFs in
        circulation or that need one page configured differently:

        num : pin the number (a campaign with historical gaps in its numbering
            must pin every study, or later ones silently renumber).
        name : pin the part after NN_, when the PDF name can't be spelled as a
            Python identifier (e.g. "test5_floor1e-2_flimscan").
        pages : replace the campaign page set for this study only.
        page_opts : {page name: {...}} patched over the campaign page options
            for this study only -- e.g. dropping a reference dataset that is
            meaningless for this study's machine.
        """
        def register(f):
            self._studies[f.__name__] = dict(
                fn=f,
                num=num if num is not None else len(self._studies) + 1,
                name=name or f.__name__,
                pages=pages,
                page_opts=page_opts or {},
            )
            return f
        return register(fn) if fn is not None else register

    @property
    def studies(self):
        return dict(self._studies)

    def slug_for(self, key):
        """PDF stem for a registered study (see `numbered`)."""
        e = self._studies[key]
        return f"{e['num']:02d}_{e['name']}" if self.numbered else e["name"]

    def listing(self):
        return ", ".join(f"{e['num']}={n}" for n, e in self._studies.items())

    # --- running ------------------------------------------------------------
    def _lookup(self, name):
        """Resolve a study selector to its registered function name.

        Tried in order: the exact name, the study number ("02" and 2 both
        work), then a unique match ignoring a leading prefix in either
        direction -- so a campaign that dropped a "study_" prefix from its
        function names still answers to the old names scattered through its
        notes and logs. Ambiguity is an error, never a silent guess.
        """
        if callable(name):
            name = name.__name__
        if name in self._studies:
            return name

        num = str(name).lstrip("0") or "0"  # "02" -> "2"; tolerate zero-pad
        by_num = {str(e["num"]): n for n, e in self._studies.items()}
        if num in by_num:
            return by_num[num]

        loose = [n for n in self._studies
                 if n.endswith(f"_{name}") or str(name).endswith(f"_{n}")]
        if len(loose) == 1:
            return loose[0]
        if len(loose) > 1:
            raise SystemExit(
                f"Ambiguous study '{name}' -- matches {sorted(loose)}. "
                f"Use the exact name or its number."
            )
        raise SystemExit(f"Unknown study '{name}'. Registered: {self.listing()}")

    def run(self, name, pages=None):
        """Build one registered study's PDF, by name, number, or function.

        `pages` overrides the campaign page set for this call only (handy when
        iterating on one page interactively -- it does not change the study).
        """
        key = self._lookup(name)
        entry = self._studies[key]
        return self.run_cases(
            slug=self.slug_for(key),
            cases=entry["fn"](),
            notes=entry["fn"].__doc__,
            pages=pages if pages is not None else entry["pages"],
            page_opts=entry["page_opts"],
        )

    def run_cases(self, slug, cases, notes="", pages=None, page_opts=None):
        """Build a PDF for an ad-hoc selection (no @study registration).

        Normally you want run() on a registered study, so the selection and its
        conclusion stay recorded in the campaign file.
        """
        cs = self.collection(cases, self.loader, palette=self.palette,
                             **self.collection_opts)
        if self.needs_dumps:
            cs = cs.load()
        ctx = PageContext(cs, slug, notes, self)
        resolved = resolve_pages(self.pages if pages is None else pages, page_opts)
        path = self.outdir / f"{slug}.pdf"
        Report(path, page_size=self.page_size).build(resolved, ctx)
        print(f"  wrote {path}")
        return path

    def run_all(self):
        """Build every registered study, isolating per-study failures.

        Same reasoning as page-level isolation in Report.build: a study whose
        cases have been deleted must not cost you the other thirty PDFs. A
        single run() still raises, so an explicit one-study run fails loudly.
        Returns {study name: path or None} and prints a summary of what broke.
        """
        results, failed = {}, {}
        for n in self._studies:
            try:
                results[n] = self.run(n)
            except Exception as e:  # noqa: BLE001
                results[n] = None
                failed[n] = f"{type(e).__name__}: {e}"
                print(f"  [study:{n}] FAILED: {failed[n]}")
        if failed:
            print(f"\n{len(failed)} of {len(self._studies)} studies failed:")
            for n, msg in failed.items():
                print(f"  {self._studies[n]['num']:02d} {n}: {msg}")
        return results

    def check(self):
        """Report which studies name case directories that don't exist.

        Reads the case-directory index only -- seconds, not the hour a full
        run_all takes -- so it is the cheap way to find studies that have
        outlived their cases before committing to a rebuild.
        """
        broken = {}
        for n, e in self._studies.items():
            coll = self.collection(e["fn"](), self.loader, palette=self.palette,
                                   **self.collection_opts)
            missing = coll.missing()
            if missing:
                broken[n] = missing
        for n, missing in broken.items():
            print(f"  {self._studies[n]['num']:02d} {n}: "
                  f"{len(missing)} unresolved")
            for m in missing:
                print(f"       {m}")
        if not broken:
            print(f"  all {len(self._studies)} studies resolve every case")
        return broken

    # --- CLI ----------------------------------------------------------------
    def cli(self, argv=None):
        """Standard campaign-script command line.

            python analysis.py                  # the LAST-defined study
            python analysis.py all              # every study
            python analysis.py <name|num> ...   # the named/numbered studies
            python analysis.py --list           # what's registered
            python analysis.py --check          # studies whose cases are gone
        """
        args = list(sys.argv[1:] if argv is None else argv)
        if not self._studies:
            raise SystemExit("No studies registered.")
        if args and args[0] in ("-c", "--check"):
            self.check()
            return
        if args and args[0] in ("-l", "--list"):
            for n, e in self._studies.items():
                slug = self.slug_for(n)
                alias = f"   -> {slug}.pdf" if slug != f"{e['num']:02d}_{n}" else ""
                print(f"  {e['num']:02d}  {n}{alias}")
            return
        if not args:
            self.run(next(reversed(self._studies)))
        elif args == ["all"]:
            self.run_all()
        else:
            for a in args:
                self.run(a)
        print("Done")
