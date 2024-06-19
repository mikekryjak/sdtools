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

def heavyplot(casename, save = True):
    """
    Computationally heavy plots
    
    Inputs
    -----

    """
    db = CaseDB(case_dir = r"/users/mjk557/scratch/cases", grid_dir = r"/users/mjk557/scratch/cases")
    
    casename = casename.split(r"/")[-1]
    case = db.load_case_2D(casename, use_squash = False)
    ds = case.ds
        
    # ds = ds2
    tlen = ds.dims["t"]
    tres = 10
    ts = np.linspace(0, tlen-1, tres, dtype = int)
    colors = [plt.cm.get_cmap("plasma", tres)(x) for x in range(tres)]

    toplot = {}
    for t in ts:
        toplot[f"t={t/10}ms"] = ds.isel(t=t, x = slice(2,-2))

    if save is True:
        save_name = f"hmon_{casename}"
    else:
        save_name = ""

    lineplot(
        toplot,
        clean_guards = False,
        params = ["Te", "Td", "Ne", "Nd", "Pd+", "Pd", "Sd+_iz"],
        regions = ["omp", "outer_lower", "field_line"],
        colors = colors,
        save_name = save_name
    )

        
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