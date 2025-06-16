#!/usr/bin/env python3

# Reading with cache for extra speed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import boutdata
from boututils.options import BOUTOptions
from boutdata.collect import create_cache
import argparse
import os, sys
import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hermes3.plotting import lineplot
from hermes3.case_db import CaseDB
from hermes3.accessors import *

import warnings
warnings.filterwarnings("ignore")

def heavyplot(casename, save = True):
    """
    Computationally heavy plots
    
    Inputs
    -----

    """
    tstart = tm.time()
    db = CaseDB(case_dir = r"/home/mike/work/cases", grid_dir = r"/home/mike/work/cases")
    
    casename = casename.split(r"/")[-1]
    case = db.load_case_2D(casename, use_squash = False)
    case.extract_2d_tokamak_geometry()
    ds = case.ds
        
    # ds = ds2
    tlen = ds.dims["t"]
    if tlen > 10:
        tres = 10
    else:
        tres = tlen
    ts = np.linspace(0, tlen-1, tres, dtype = int)
    colors = [plt.cm.get_cmap("Spectral_r", tres)(x) for x in range(tres)]

    toplot = {}
    for t in ts:
        toplot[f"t={t*1e-3}ms"] = ds.isel(t=t, x = slice(2,-2))

    if save is True:
        save_name = f"hmon_{casename}"
    else:
        save_name = ""

    lineplot(
        toplot,
        clean_guards = False,
        params = ["Te", "Td",  "Ne", "Nd", "Rd+_ex"],
        regions = ["omp", "outer_lower", "field_line"],
        colors = colors,
        save_name = save_name
    )

    tend = tm.time()
    print(f"Executed in {tend-tstart:.1f} seconds")
        
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------

if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Case monitor")
    parser.add_argument("path", type=str, help = "Path to case")
    # parser.add_argument("-p", action="store_true", help = "Plot?")
    # parser.add_argument("-t", action="store_true", help = "Table?")
    # parser.add_argument("-s", action="store_true", help = "Save figure?")
    # parser.add_argument("-neutrals", action="store_true", help = "Neutral focused plot?")
    
    # Extract arguments and call function
    args = parser.parse_args()
    heavyplot(args.path)