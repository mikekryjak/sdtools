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
    def __init__(self, path):
        
        bal = self.bal = nc.Dataset(os.path.join(path, "balance.nc"))
        g = self.g = read_b2fgmtry(where=path)
        self.params = list(self.bal.variables.keys())
        
        # Get cell centre coordinates
        self.g["R"] = np.mean(g["crx"], axis=2)
        self.g["Z"] = np.mean(g["cry"], axis=2)
        
        # Set up geometry
        
        omp = int((g["rightcut"][0] + g["rightcut"][1])/2) + 1
        imp = int((g["leftcut"][0] + g["leftcut"][1])/2) + 1
        upper_break = int(imp + (omp - imp)/2) - 2
        sep = min(g["topcut"][0], g["topcut"][1]) +1
        
        self.g["sep"] = sep
        self.g["imp"] = imp
        self.g["upper_break"] = upper_break
        self.g["sep"] = sep

        # poloidal, radial, corners
        # p = [imp, slice(None,None), 0]

        s = {} # slices
        s["imp"] = [imp, slice(None,None)]
        s["omp"] = [omp, slice(None,None)]
        s["outer"] = [slice(upper_break,None), sep]
        s["outer_lower"] = [slice(omp,None), sep]
        s["outer_upper"] = [slice(upper_break, omp), sep]
        s["inner"] = [slice(None, upper_break-1), sep]
        s["inner_lower"] = [slice(None, imp+1), sep]
        s["inner_upper"] = [slice(imp, upper_break-1), sep]
        self.s = s
        
        # bal.close()
        
    def close(self):
        self.bal.close()
        
    def plot_2d(self, param,
             ax = None,
             fig = None,
             norm = None,
             data = np.array([]),
             cmap = "Spectral_r",
             antialias = False,
             linecolor = "k",
             linewidth = 0,
             vmin = None,
             vmax = None,
             logscale = False,
             alpha = 1,
             separatrix = True):
        
        if len(data)==0:
            data = self.bal[param][:]
        
        if vmin == None:
            vmin = data.min()
        if vmax == None:
            vmax = data.max()
        if norm == None:
            norm = create_norm(logscale, norm, vmin, vmax)
        if ax == None:
            fig, ax = plt.subplots(dpi = 150)
        

        # Following SOLPS convention: X poloidal, Y radial
        crx = self.bal["crx"]
        cry = self.bal["cry"]
        
        Nx = crx.shape[2]
        Ny = crx.shape[1]
        
        print(data.shape)
        print(Nx, Ny)
        # print(cry.shape)
        
        
        # In hermes-3 and needed for plot: lower left, lower right, upper right, upper left, lower left
        # SOLPS crx structure: lower left, lower right, upper left, upper right
        # So translating crx is gonna be 0, 1, 3, 2, 0
        # crx is [corner, Y(radial), X(poloidal)]
        idx = [np.array([0, 1, 3, 2, 0])]

        # Make polygons
        patches = []
        for i in range(Nx):
            for j in range(Ny):
                p = mpl.patches.Polygon(
                    np.concatenate([crx[:,j,i][tuple(idx)], cry[:,j,i][tuple(idx)]]).reshape(2,5).T,
                    
                    fill=False,
                    closed=True,
                )
                patches.append(p)
                
        # Polygon colors
        colors = data.transpose().flatten()
        polys = mpl.collections.PatchCollection(
            patches, alpha = alpha, norm = norm, cmap = cmap, 
            antialiaseds = antialias,
            edgecolors = linecolor,
            linewidths = linewidth,
            joinstyle = "bevel")

        polys.set_array(colors)
        
        if fig != None:
            fig.colorbar(polys)
        ax.add_collection(polys)
        ax.set_aspect("equal")
        
        if separatrix is True:
            self.plot_separatrix(ax = ax)
            
    def plot_separatrix(self, ax):

        # Author: Matteo Moscheni
        # E-mail: matteo.moscheni@tokamakenergy.co.uk
        # February 2022

        # try:    b2fgmtry = load_pickle(where = where, verbose = False, what = "b2fgmtry")
        # except: b2fgmtry = read_b2fgmtry(where = where, verbose = False, save = True)
        b2fgmtry = self.g
        colour = "white"

        iy = int(b2fgmtry['ny'] / 2)

        if len(b2fgmtry['rightcut']) == 2:
            ix_mid = int((b2fgmtry['rightcut'][1] + b2fgmtry['leftcut'][1]) / 2 - 1)

            for ix in range(ix_mid - 2):
                x01 = [b2fgmtry['crx'][ix,iy,0], b2fgmtry['crx'][ix,iy,1]]
                y01 = [b2fgmtry['cry'][ix,iy,0], b2fgmtry['cry'][ix,iy,1]]
                ax.plot(x01, y01, c = colour, lw = 2)

            for ix in range(ix_mid, b2fgmtry['nx']):
                x01 = [b2fgmtry['crx'][ix,iy,0], b2fgmtry['crx'][ix,iy,1]]
                y01 = [b2fgmtry['cry'][ix,iy,0], b2fgmtry['cry'][ix,iy,1]]
                ax.plot(x01, y01, c = colour, lw = 2)
        else:
            for ix in range(b2fgmtry['nx']):
                x01 = [b2fgmtry['crx'][ix,iy,0], b2fgmtry['crx'][ix,iy,1]]
                y01 = [b2fgmtry['cry'][ix,iy,0], b2fgmtry['cry'][ix,iy,1]]
                ax.plot(x01, y01, c = colour, lw = 2)

        return
        
    def get_1d_radial_data(
        self,
        param,
        region = "omp",
        verbose = False
    ):
        """
        Returns OMP or IMP data from the balance file
        Note that the balance file shape is a transpose of the geometry shape (e.g. crx)
        and contains guard cells.        
        """
        
        if any([region in name for name in ["omp", "imp"]]):
            p = self.s[region] 
        else:
            raise Exception(f"Unrecognised region: {region}")
    
        df = pd.DataFrame()
        df["dist"] = self.g["R"][p[0], p[1]] - self.g["R"][p[0], self.g["sep"]] 
        df["R"] = self.g["R"][p[0], p[1]]
        df["Z"] = self.g["Z"][p[0], p[1]]
        df[param] = self.bal[param][:].transpose()[p[0], 1:-1] # Drop guard cells
        
        return df
        
        
def create_norm(logscale, norm, vmin, vmax):
    if logscale:
        if norm is not None:
            raise ValueError(
                "norm and logscale cannot both be passed at the same time."
            )
        if vmin * vmax > 0:
            # vmin and vmax have the same sign, so can use standard log-scale
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            # vmin and vmax have opposite signs, so use symmetrical logarithmic scale
            if not isinstance(logscale, bool):
                linear_scale = logscale
            else:
                linear_scale = 1.0e-5
            linear_threshold = min(abs(vmin), abs(vmax)) * linear_scale
            if linear_threshold == 0:
                linear_threshold = 1e-4 * vmax   # prevents crash on "Linthresh must be positive"
            norm = mpl.colors.SymLogNorm(linear_threshold, vmin=vmin, vmax=vmax)
    elif norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return norm