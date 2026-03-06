#!/usr/bin/env python3

# Reading with cache for extra speed
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


def plot_residuals_2d(
    casename,
    mode="frequency",
    vmin=None,
    vmax=None,
    ylim=None
    variables=["Nd+", "Pd+", "Pe", "NVd+", "Nd", "Pd", "NVd"],
    save=True,
):
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

    def get_ddt(var, mode):
        if mode == "standard":
            return ds[f"ddt({var})"]
        elif mode == "frequency":
            return ds[f"ddt({var})"] / ds[var]

    toplot = []

    for var in variables:
        toplot.append(
            dict(
                data=ds[var],
                title=f"ddt({var}) ({mode})",
                vmin = vmin,
                vmax = vmax
            )
        )

    if save:
        save_name = f"residuals_{casename}"
    else:
        save_name = None

    plot_residuals_2d(
        toplot,
        ylim=ylim,
        title=f"Residuals",
        clean_guards=True,
        separatrix=True,
        save_path=save_name,
        cmap="Spectral_r",
    )

    tend = tm.time()
    print(f"Executed in {tend - tstart:.1f} seconds")


# ------------------------------------------------------------
# PARSER
# ------------------------------------------------------------

if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description="Case monitor")
    parser.add_argument("path", type=str, help="Path to case")
    parser.add_argument("--mode", type=str, default="frequency", help="Mode for ddt calculation (standard or frequency)")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum value for color scale")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for color scale")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, help="Y-axis limits for plot")
    parser.add_argument("--variables", type=str, nargs="+", default=["Nd+", "Pd+", "Pe", "NVd+", "Nd", "Pd", "NVd"], help="Variables to plot")
    parser.add_argument("--save", action="store_true", help="Save figure?")

    # Extract arguments and call function
    args = parser.parse_args()
    plot_residuals_2d(args.path, mode=args.mode, vmin=args.vmin, vmax=args.vmax, ylim=args.ylim, variables=args.variables, save=args.save)
