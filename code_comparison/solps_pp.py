from boututils.datafile import DataFile
from boutdata.collect import collect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pathlib
import platform
import traceback
import xarray as xr
import xbout
import scipy
import re
import netCDF4 as nc
import matplotlib as mpl

onedrive_path = onedrive_path = str(os.getcwd()).split("OneDrive")[0] + "OneDrive"
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\gridtools"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\sdtools"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\soledge"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages"))


# from gridtools.hypnotoad_tools import *
from gridtools.b2_tools import *
from gridtools.utils import *

from hermes3.case_db import *
from hermes3.load import *
from hermes3.named_selections import *
from hermes3.plotting import *
from hermes3.grid_fields import *
from hermes3.accessors import *
from hermes3.utils import *
from code_comparison.viewer_2d import *
from code_comparison.code_comparison import *

import gridtools.solps_python_scripts.setup
from gridtools.solps_python_scripts.plot_solps       import plot_1d, plot_2d, plot_wall_loads
from gridtools.solps_python_scripts.read_ft44 import read_ft44
import ipywidgets as widgets

from solps_python_scripts.read_b2fgmtry import read_b2fgmtry



class SOLPScase():
    def __init__(path):
        bal = self.bal = nc.Dataset(os.path.join(path, "balance.nc"))
        g = self.g = read_b2fgmtry(where=path)
        
        # Set up geometry
        omp = int((g["rightcut"][0] + g["rightcut"][1])/2) + 1
        imp = int((g["leftcut"][0] + g["leftcut"][1])/2) + 1
        upper_break = int(imp + (omp - imp)/2) - 2
        sep = min(g["topcut"][0], g["topcut"][1]) +1

        crx = g["crx"]
        cry = g["cry"]
        fig, ax = plt.subplots(dpi = 160)

        # poloidal, radial, corners
        # p1 = [imp, slice(None,None), 0]

        sol_ring = 1

        s = {} # slices
        s["imp"] = [imp, slice(None,None), 0]
        s["omp"] = [omp, slice(None,None), 0]
        s["outer"] = [slice(upper_break,None), sep + sol_ring, 0]
        s["outer_lower"] = [slice(omp,None), sep + sol_ring, 0]
        s["outer_upper"] = [slice(upper_break, omp), sep + sol_ring, 0]
        s["inner"] = [slice(None, upper_break-1), sep + sol_ring, 0]
        s["inner_lower"] = [slice(None, imp+1), sep + sol_ring, 0]
        s["inner_upper"] = [slice(imp, upper_break-1), sep + sol_ring, 0]