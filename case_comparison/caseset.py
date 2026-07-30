"""CaseSet -- the case selection a study compares, with everything a page needs.

One identity per case: the simulation DIRECTORY NAME is the key everywhere (in
the study definition, in the load cache, in the provenance lookups). A page
asks a CaseSet for datasets keyed by display label; it never loads anything
itself.
"""

import numpy as np

from . import provenance

# Auto-assigned line colours when a case doesn't override color=. The first
# case in a set is always black instead (see CaseSet.__init__) -- it's normally
# the baseline every other case is judged against, so it should read as the
# fixed reference line, not just another palette colour.
DEFAULT_PALETTE = ["teal", "darkorange", "firebrick", "limegreen", "deeppink",
                   "royalblue", "darkviolet", "olive"]


class MissingCase(KeyError):
    """A study names a case directory that doesn't exist under case_root."""


class CaseLoader:
    """Loads and caches Hermes-3 cases by directory name.

    One loader per campaign, shared by every study, so a case appearing in
    several studies is read from disk once (this is what makes
    `ipython -i analysis.py` cheap to iterate in).
    """

    def __init__(self, case_root, grid_root=None, load_kwargs=None,
                 dir_fallback=None):
        self.case_root = case_root
        self.grid_root = grid_root or case_root
        self.load_kwargs = dict(use_squash=False, verbose=True)
        if load_kwargs:
            self.load_kwargs.update(load_kwargs)
        # Second place to resolve a case name, tried only when no directory
        # exists under case_root: a callable name -> path, or None. A campaign
        # that keeps an archive of finished runs passes its lookup in here, so
        # a study outlives the simulation output it was built from. It holds
        # copied text -- logs, settings -- not datasets, so it answers dir()
        # and everything provenance reads, and never case().
        #
        # Deliberately a plain callable rather than an archive type: what such
        # an archive contains is the campaign's business, not this package's.
        self.dir_fallback = dir_fallback
        self._db = None
        self._cases = {}       # name -> sdtools Case
        self._records = {}     # name -> light record, or None if unloadable
        self._hermesdata = {}  # name -> code_comparison Hermesdata (lazy)

    @property
    def db(self):
        if self._db is None:
            from hermes3.case_db import CaseDB
            self._db = CaseDB(case_dir=self.case_root, grid_dir=self.grid_root)
        return self._db

    def case(self, name):
        """The sdtools Case, loaded on first use and cached thereafter."""
        if name not in self._cases:
            if name not in self.db.casepaths:
                # CaseDB would raise a bare KeyError on the directory name,
                # which says nothing about where it looked. A study outliving
                # its case directory is common enough to be worth naming.
                if self._fallback_dir(name) is not None:
                    raise MissingCase(
                        f"case '{name}' has no simulation output under "
                        f"{self.case_root} -- only an archived record, which "
                        f"holds text and not datasets. A page asked for its "
                        f"dataset, so either that page cannot be drawn for "
                        f"this case or the run must be repeated."
                    )
                raise MissingCase(
                    f"case directory '{name}' not found under {self.case_root} "
                    f"(a study can outlive its cases -- check the name, or "
                    f"whether the run was deleted/renamed)"
                )
            case = self.db.load_case_2D(name, **self.load_kwargs)
            case.extract_2d_tokamak_geometry()
            self._cases[name] = case
        return self._cases[name]

    def _fallback_dir(self, name):
        """Where dir_fallback says `name` lives, or None. Never raises: a
        broken lookup must degrade to "no fallback", not kill the report."""
        if self.dir_fallback is None:
            return None
        try:
            return self.dir_fallback(name)
        except Exception as e:  # noqa: BLE001
            print(f"  [fallback lookup failed] {name}: {e}")
            return None

    def missing(self, names):
        """Which of `names` can be resolved in neither place. Cheap -- reads
        the directory index only, never opens a dataset."""
        return [n for n in names
                if n not in self.db.casepaths and self._fallback_dir(n) is None]

    def record(self, name):
        """A LIGHT reduction of a case: final-time slice + the 1-D t and wtime
        series, and nothing else.

        For comparisons spanning many cases (a scan campaign loads whole series
        at a time), holding every full time-resolved 2-D field would not fit.
        Returns None -- cached, so a broken case is only attempted once -- if
        the case can't be loaded or has no usable output, since one failed run
        must not kill a whole series.
        """
        if name in self._records:
            return self._records[name]
        try:
            ds = self.case(name).ds
            if "t" not in ds.dims or ds.sizes.get("t", 0) < 1:
                raise ValueError("no time output")
            self._records[name] = dict(
                name=name,
                last=ds.isel(t=-1).load(),
                t_ms=np.asarray(ds["t"].values, dtype=float) * 1000.0,
                wtime=np.asarray(ds["wtime"].values, dtype=float),
            )
        except Exception as e:  # noqa: BLE001 -- a failed run must not stop the study
            print(f"  [load failed] {name}: {e}")
            self._records[name] = None
        return self._records[name]

    def forget(self, name):
        """Drop the full Case for `name`, keeping any light record.

        Scan pages only ever need records; releasing the full dataset after
        reduction keeps a many-case study's memory flat.
        """
        self._cases.pop(name, None)

    def ds(self, name):
        return self.case(name).ds

    def hermesdata(self, name):
        """Last-timestep case wrapped as a code_comparison Hermesdata.

        Only what lineplot_compare (the SOLPS/SOLEDGE comparison stack) needs.
        Imported lazily and cached: code_comparison hard-imports the SOLEDGE
        wrapper, which pulls in tkinter, so a campaign that never asks for this
        never pays for it.
        """
        if name not in self._hermesdata:
            from code_comparison.code_comparison import Hermesdata
            ds = self.ds(name)
            if "t" in ds.dims:
                ds = ds.isel(t=-1)
            hd = Hermesdata()
            hd.read_case(ds.load())
            self._hermesdata[name] = hd
        return self._hermesdata[name]

    def dir(self, name):
        """Where to read this case's files from, as a str, or None if unknown.

        The simulation directory when it still exists, otherwise whatever
        dir_fallback offers. Provenance and option-diff read text files by
        path, so they keep working off an archived record once the simulation
        output is gone -- which is the point of having one.
        """
        d = self.db.casepaths.get(name)
        if d is not None:
            return str(d)
        d = self._fallback_dir(name)
        return str(d) if d is not None else None


class CaseSet:
    """One study's ordered case selection: names, labels, colours, data access.

    Built from the dict a study function returns -- {name: "label"} or
    {name: dict(label=, color=)} -- keyed by simulation directory name.
    Colour is auto-assigned by position unless a case overrides it: first case
    black (the baseline), later cases take palette colours in order.
    """

    def __init__(self, cases, loader, palette=None):
        palette = palette or DEFAULT_PALETTE
        self.loader = loader
        self.specs = {}
        for i, (name, spec) in enumerate(cases.items()):
            d = dict(spec) if isinstance(spec, dict) else dict(label=str(spec))
            d.setdefault("label", name)
            d.setdefault("color",
                         "black" if i == 0 else palette[(i - 1) % len(palette)])
            self.specs[name] = d

    # --- identity -----------------------------------------------------------
    @property
    def names(self):
        return list(self.specs)

    def label(self, name):
        return self.specs[name]["label"]

    def color(self, name):
        return self.specs[name]["color"]

    def __len__(self):
        return len(self.specs)

    def __iter__(self):
        """Iterate (name, spec) pairs in study order."""
        return iter(self.specs.items())

    # --- data, keyed by display label (what plotting functions want) --------
    def datasets(self):
        """{label: dataset}, study order. Loads any case not yet in memory."""
        return {self.label(n): self.loader.ds(n) for n in self.names}

    def colors(self):
        """{label: colour}, study order."""
        return {self.label(n): self.color(n) for n in self.names}

    def color_list(self):
        """[colour], study order -- for plotters taking a positional list."""
        return [self.color(n) for n in self.names]

    def hermesdata(self):
        """{name: Hermesdata} for the whole set (see CaseLoader.hermesdata)."""
        return {n: self.loader.hermesdata(n) for n in self.names}

    def load(self):
        """Force every case in, so a page failing later doesn't half-load."""
        for n in self.names:
            self.loader.case(n)
        return self

    # --- provenance ---------------------------------------------------------
    def dirs(self):
        """{name: case directory}, study order."""
        return {n: self.loader.dir(n) for n in self.names}

    def check_levels(self):
        """{name: CHECK level as str}."""
        return {n: provenance.check_level(self.loader.dir(n)) for n in self.names}

    def missing(self):
        """Case names with no directory under case_root. Cheap -- reads the
        directory index only, so it is the pre-flight check (Campaign.check)."""
        return self.loader.missing(self.names)

    def has_var(self, var):
        """True iff EVERY case's dataset contains `var`.

        Used by pages to drop panels for diagnostics some cases didn't save
        (e.g. ddt() fields need diagnose=true under [solver] at run time).
        """
        return all(var in self.loader.ds(n) for n in self.names)
