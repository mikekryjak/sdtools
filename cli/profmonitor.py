#!/usr/bin/env python3

# Reading with cache for extra speed
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os, sys
import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hermes3.plotting import lineplot
from hermes3.case_db import CaseDB
from hermes3.accessors import *

import warnings

warnings.filterwarnings("ignore")


def _is_case_path(path):
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "BOUT.inp"))
        or bool(glob.glob(os.path.join(path, "BOUT.dmp*")))
    )


def _expand_case_paths(path):
    if isinstance(path, (str, os.PathLike)):
        raw_paths = [os.fspath(path)]
    else:
        raw_paths = [os.fspath(item) for item in path]

    case_paths = []
    seen = set()

    for raw_path in raw_paths:
        matches = (
            sorted(glob.glob(raw_path)) if glob.has_magic(raw_path) else [raw_path]
        )
        matches = [os.path.normpath(match) for match in matches if _is_case_path(match)]

        if not matches:
            raise FileNotFoundError(f"No case directories match '{raw_path}'")

        for match in matches:
            if match not in seen:
                seen.add(match)
                case_paths.append(match)

    return case_paths


def heavyplot(path, save=True):
    case_paths = _expand_case_paths(path)

    for index, case_path in enumerate(case_paths):
        if index > 0:
            print()

        _heavyplot_case(case_path, save=save)


def _heavyplot_case(casename, save=True):
    """
    Computationally heavy plots

    Inputs
    -----

    """
    tstart = tm.time()
    db = CaseDB(case_dir=r"/home/mike/work/cases", grid_dir=r"/home/mike/work/cases")
    casename = casename.split(r"/")[-1]
    case = db.load_case_2D(casename, use_squash=False)
    case.extract_2d_tokamak_geometry()
    ds = case.ds

    tlen = ds.sizes["t"]
    if tlen > 10:
        tres = 10
    else:
        tres = tlen

    ts = np.linspace(0, tlen - 1, tres, dtype=int)

    colors = [plt.cm.get_cmap("Spectral_r", tres)(x) for x in range(tres)]

    toplot = {}
    for t in ts:
        time_ms = ds.isel(t=t)["t"].values * 1e3
        toplot[f"t={time_ms:.3f}ms"] = ds.isel(t=t, x=slice(2, -2))

    if save is True:
        save_name = f"mon_prof_{casename}"
    else:
        save_name = ""

    lineplot(
        toplot,
        params=["Te", "Td", "Ne", "Nd", "NVd+", "NVd"],
        regions=["omp", "outer_lower_target", "field_line"],
        colors=colors,
        save_name=save_name,
        combine_regions=True,
    )

    if save is True:
        print(f"Saved plot to {os.path.abspath(save_name)}.png")

    tend = tm.time()
    print(f"Executed in {tend - tstart:.1f} seconds")


# ------------------------------------------------------------
# PARSER
# ------------------------------------------------------------

if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description="Case monitor")
    parser.add_argument("paths", nargs="+", help="Case path(s) or wildcard pattern(s)")
    # parser.add_argument("-p", action="store_true", help = "Plot?")
    # parser.add_argument("-t", action="store_true", help = "Table?")
    # parser.add_argument("-s", action="store_true", help = "Save figure?")
    # parser.add_argument("-neutrals", action="store_true", help = "Neutral focused plot?")

    # Extract arguments and call function
    args = parser.parse_args()
    heavyplot(args.paths)
