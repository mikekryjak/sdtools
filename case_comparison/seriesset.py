"""SeriesSet -- the scan selection a scan study compares.

Where a CaseSet is a handful of named cases compared in full detail, a SeriesSet
is several SCANS compared as lines: each series is a set of runs differing only
in one swept parameter, named by a GLOB rather than spelled out case by case,
with the sweep coordinate parsed from each case name. Every case is reduced to a
few scalars, so a study can span dozens of runs.

The parts that differ per campaign are injected, not hardcoded: the coordinate
pattern (`x_pattern`, `x_label`) and the scalars (`scalar_fn`).
"""

import re
import fnmatch

import numpy as np

from . import provenance
from .caseset import DEFAULT_PALETTE

# Auto-assigned marker per series. Colour alone hides a series whose line
# coincides with another's; distinct marker shapes keep both legible.
DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


class SeriesSet:
    """One scan study's ordered series selection.

    Built from {label: "glob"} or {label: dict(glob=, color=, marker=,
    exclude=)}. A glob may also be a LIST of patterns (a case matches if it
    matches any), so one series can pull cases from more than one naming run.

    Parameters
    ----------
    series : the study's series dict (above).
    loader : CaseLoader, shared with the campaign.
    palette, markers : auto-assigned styling; first series is black.
    x_pattern : regex with ONE capture group giving the sweep coordinate from a
        case name, e.g. r"flim([0-9]+\\.?[0-9]*)". A case whose name doesn't
        match is dropped with a warning -- it has no position on the axis.
    x_label : axis label for that coordinate.
    x_name : SHORT prose name for it, used in sentences (page titles) where
        the full axis label -- which may carry units or mathtext -- reads
        badly and changes the layout. Defaults to x_label.
    exclude : glob patterns dropped from EVERY series (stale/superseded run
        directories a series glob would otherwise sweep up).
    scalar_fn : callable(record) -> {key: value}, the campaign's physics
        scalars. Performance scalars are always computed.
    perf_metric : which variability scalar (see scalars.perf_scalars).
    """

    def __init__(self, series, loader, palette=None, markers=None,
                 x_pattern=r"flim([0-9]+\.?[0-9]*)", x_label="Flux limiter",
                 x_name=None, exclude=(), scalar_fn=None,
                 perf_metric="p95_over_median"):
        palette = palette or DEFAULT_PALETTE
        markers = markers or DEFAULT_MARKERS
        self.loader = loader
        self.x_re = re.compile(x_pattern)
        self.x_label = x_label
        self.x_name = x_name or x_label
        self.global_exclude = list(exclude)
        self.scalar_fn = scalar_fn
        self.perf_metric = perf_metric

        self.specs = {}
        for i, (label, spec) in enumerate(series.items()):
            d = dict(spec) if isinstance(spec, dict) else dict(glob=str(spec))
            if "glob" not in d:
                raise ValueError(f"series '{label}' has no glob pattern")
            d.setdefault("color",
                         "black" if i == 0 else palette[(i - 1) % len(palette)])
            d.setdefault("marker", markers[i % len(markers)])
            self.specs[label] = d

        self.matched = {}   # label -> [(case name, x), ...] sorted by x
        self.scalars = {}   # case name -> {key: value}
        self.perf = {}      # case name -> perf scalars + check level

    # --- identity -----------------------------------------------------------
    @property
    def labels(self):
        return list(self.specs)

    def color(self, label):
        return self.specs[label]["color"]

    def marker(self, label):
        return self.specs[label]["marker"]

    def __len__(self):
        return len(self.specs)

    def names(self):
        """Every matched case name, across all series, in series order."""
        return [n for label in self.labels for n, _ in self.matched.get(label, [])]

    # --- discovery ----------------------------------------------------------
    def parse_x(self, name):
        """Sweep coordinate from a case name, or None if the token is absent."""
        m = self.x_re.search(name)
        return float(m.group(1)) if m else None

    def match(self, label):
        """Resolve one series' glob(s) to sorted [(name, x), ...]."""
        spec = self.specs[label]
        glob = spec["glob"]
        patterns = [glob] if isinstance(glob, str) else list(glob)
        excludes = [*self.global_exclude, *spec.get("exclude", [])]

        hits = []
        for name in self.loader.db.casepaths:
            if not any(fnmatch.fnmatch(name, p) for p in patterns):
                continue
            if any(fnmatch.fnmatch(name, ex) for ex in excludes):
                print(f"  [exclude] {name}")
                continue
            x = self.parse_x(name)
            if x is None:
                print(f"  [skip] {name}: name carries no sweep coordinate")
                continue
            hits.append((name, x))
        hits.sort(key=lambda nx: nx[1])
        return hits

    def missing(self):
        """Series whose glob matches no case on disk, as readable strings.

        A scan study can't have a "missing case" the way a case study does --
        its cases are discovered, not named -- so the failure mode is a glob
        that has stopped matching (renamed runs, a deleted series). Reads the
        directory index only, so it is cheap enough for Campaign.check.
        """
        out = []
        for label in self.labels:
            if not self.match(label):
                glob = self.specs[label]["glob"]
                glob_str = glob if isinstance(glob, str) else " + ".join(glob)
                out.append(f"series '{label}': no case matches {glob_str}")
        return out

    def load(self):
        """Resolve every series, reduce each case to scalars, drop failures.

        Cases that fail to load are dropped from their series (a dead run must
        not remove the rest of the scan), and the full dataset is released as
        soon as its scalars are computed -- see CaseLoader.forget.
        """
        from .scalars import perf_scalars

        self.matched = {label: self.match(label) for label in self.labels}
        for label, pts in self.matched.items():
            print(f"  {label}: {len(pts)} cases -> "
                  f"{[f'{x:g}' for _, x in pts]}")

        for label, pts in self.matched.items():
            kept = []
            for name, x in pts:
                rec = self.loader.record(name)
                if rec is None:
                    continue
                if self.scalar_fn is not None:
                    self.scalars[name] = self.scalar_fn(rec)
                p = perf_scalars(rec, metric=self.perf_metric)
                p["check"] = provenance.check_level(self.loader.dir(name))
                self.perf[name] = p
                self.loader.forget(name)  # scalars extracted; release the data
                kept.append((name, x))
            self.matched[label] = kept
        return self

    # --- what pages plot ----------------------------------------------------
    def points(self, value_of):
        """{label: [(x, value), ...]} for a per-case value function.

        `value_of` maps a case name to a number; non-finite values are dropped
        so a line skips that point instead of breaking. Points come back sorted
        by x, so a plot connects them low -> high along the scan.
        """
        out = {}
        for label in self.labels:
            xy = []
            for name, x in self.matched.get(label, []):
                v = value_of(name)
                if v is not None and np.isfinite(v):
                    xy.append((x, float(v)))
            xy.sort()
            out[label] = xy
        return out

    def scalar_points(self, key):
        """points() for one physics scalar."""
        return self.points(lambda n: self.scalars.get(n, {}).get(key, np.nan))

    def perf_points(self, key):
        """points() for one performance scalar."""
        return self.points(lambda n: self.perf.get(n, {}).get(key, np.nan))

    def paired_points(self, x_key, y_key):
        """{label: [(x_value, y_value, sweep coordinate), ...]}.

        For the trade-off page, where BOTH axes are scalars and the sweep
        coordinate becomes a point label rather than an axis.
        """
        out = {}
        for label in self.labels:
            pts = []
            for name, x in self.matched.get(label, []):
                xv = self.scalars.get(name, {}).get(x_key, np.nan)
                yv = self.perf.get(name, {}).get(y_key, np.nan)
                if np.isfinite(xv) and np.isfinite(yv):
                    pts.append((x, float(xv), float(yv)))
            pts.sort()  # connect low -> high sweep so the line tracks the scan
            out[label] = pts
        return out

    # --- provenance ---------------------------------------------------------
    def dirs(self):
        return {n: self.loader.dir(n) for n in self.names()}

    def option_values(self):
        """{series label: {option: shared value or 'varies'}} over ALL options.

        One column per series, collapsed from its cases with the SAME reader
        and the same collapse rule a case campaign uses -- so a scan cover and
        a case cover can never disagree about what an option was. Deliberately
        unfiltered: the caller diffs these with provenance.diff_option_sets,
        which compares everything and only ORDERS by the campaign's key list.
        """
        out = {}
        for label in self.labels:
            names = [n for n, _ in self.matched.get(label, [])]
            if not names:
                continue
            out[label] = provenance.collapse_options(
                provenance.options_used(self.loader.dir(n)) for n in names
            )
        return out
