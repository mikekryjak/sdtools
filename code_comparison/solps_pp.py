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
# from gridtools.solps_python_scripts.plot_solps       import plot_1d, plot_2d, plot_wall_loads
# from gridtools.solps_python_scripts.read_ft44 import read_ft44
import ipywidgets as widgets

# from solps_python_scripts.read_b2fgmtry import read_b2fgmtry



class SOLPScase():
    def __init__(self, path):
        
        """
        Note that everything in the balance file is in the Y, X convention unlike the
        X, Y convention in the results. This is not intended and it reads as X, Y in MATLAB.
        You MUST transpose everything so that it's Y,X (large then small number) so it's consistent with geometry
        which is the opposite
        """
        
        raw_balance = nc.Dataset(os.path.join(path, "balance.nc"))
        
        ## Need to transpose to get over the backwards convention compared to MATLAB
        bal = {}
        for key in raw_balance.variables:
            bal[key] = raw_balance[key][:].transpose()
            
        self.bal = bal
        
        raw_balance.close()
            
            
        # g = self.g = read_b2fgmtry(where=path)
        
        self.params = list(bal.keys())
        self.params.sort()
        
        # Set up geometry
        
        g = self.g = {}
        
        # Get cell centre coordinates
        R = self.g["R"] = np.mean(bal["crx"], axis=0)
        Z = self.g["Z"] = np.mean(bal["cry"], axis=0)
        self.g["hx"] = bal["hx"]
        self.g["hy"] = bal["hy"]
        self.g["crx"] = bal["crx"]
        self.g["cry"] = bal["cry"]
        self.g["nx"] = self.g["crx"].shape[2]
        self.g["ny"] = self.g["crx"].shape[1]
        
        self.g["Btot"] = bal["bb"][3]
        self.g["Bpol"] = bal["bb"][0]
        self.g["Btor"] = np.sqrt(self.g["Btot"]**2 - self.g["Bpol"]**2) 
        
        
        
        leftix = bal["leftix"]+1
        leftix_diff = np.diff(leftix[:,1])
        g["xcut"] = xcut = np.argwhere(leftix_diff<0).squeeze()
        g["leftcut"] = [xcut[0]-2, xcut[1]+2]
        g["rightcut"] = [xcut[4]-1, xcut[3]-1]
        
        omp = self.g["omp"] = int((g["rightcut"][0] + g["rightcut"][1])/2) + 1
        imp = self.g["imp"] = int((g["leftcut"][0] + g["leftcut"][1])/2)
        upper_break = self.g["upper_break"] = g["xcut"][2]
        sep = self.g["sep"] = bal["jsep"][0] + 2
        
        # poloidal, radial, corners
        # p = [imp, slice(None,None), 0]

        s = self.s = {}
        s["imp"] = (imp, slice(None,None))
        s["omp"] = (omp, slice(None,None))
        s["outer"] = (slice(upper_break,None), sep)
        s["outer_lower"] = (slice(omp,None), sep)
        s["outer_upper"] = (slice(upper_break, omp), sep)
        s["inner"] = (slice(None, upper_break-1), sep)
        s["inner_lower"] = (slice(None, imp+1), sep)
        s["inner_upper"] = (slice(imp, upper_break-1), sep)
        
        
        
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
            data = self.bal[param]
        
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
        nx, ny = self.g["nx"], self.g["ny"]
        
        
        print(f"Data shape: {data.shape}")
        print(f"Grid shape: {nx, ny}")
        # print(cry.shape)
        
        
        # In hermes-3 and needed for plot: lower left, lower right, upper right, upper left, lower left
        # SOLPS crx structure: lower left, lower right, upper left, upper right
        # So translating crx is gonna be 0, 1, 3, 2, 0
        # crx is [corner, Y(radial), X(poloidal)]
        idx = [np.array([0, 1, 3, 2, 0])]

        # Make polygons
        patches = []
        for i in range(nx):
            for j in range(ny):
                p = mpl.patches.Polygon(
                    np.concatenate([crx[:,j,i][tuple(idx)], cry[:,j,i][tuple(idx)]]).reshape(2,5).T,
                    
                    fill=False,
                    closed=True,
                )
                patches.append(p)
                
        # Polygon colors
        colors = data.flatten()
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
        
        ## Somehow autoscale breaks sometimes
        xmin, xmax = crx.min(), crx.max()
        ymin, ymax = cry.min(), cry.max()
        xspan = xmax - xmin
        yspan = ymax - ymin

        ax.set_xlim(xmin - xspan*0.05, xmax + xspan*0.05)
        ax.set_ylim(ymin - yspan*0.05, ymax + yspan*0.05)
        
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(param)
        
        if separatrix is True:
            self.plot_separatrix(ax = ax)
            
    def plot_separatrix(self, ax, lw = 1, c = "white", ls = "-"):

        R = self.g["crx"][0,:,:]
        Z = self.g["cry"][0,:,:]
        ax.plot(R[self.s["inner"]], Z[self.s["inner"]], c, lw = lw, ls = ls)
        ax.plot(R[self.s["outer"]], Z[self.s["outer"]], c, lw = lw, ls = ls)
        
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
        df[param] = self.bal[param][p[0], 1:-1] # Drop guard cells
        
        return df
    
    def get_1d_poloidal_data(
        self,
        param,
        region = "outer_lower",
        sepadd = 0
        
    ):
        """
        Returns field line data from the balance file
        Note that the balance file shape is a transpose of the geometry shape (e.g. crx)
        and contains guard cells, so R and Z are incorrect because they don't have guards
        R and Z are provided for checking field line location only.
        only outer_lower region is provided at the moment.
        sepadd is the ring index number with sepadd = 0 being the separatrix
        
        """
        
        yind = self.g["sep"] + sepadd   # Ring index
        omp = self.g["omp"]
        hx = self.bal["hx"]
        
        if region == "outer_lower":
            selector = (yind, slice(omp, None))
        else:
            raise Exception("Unrecognised region")
        
        
        df = pd.DataFrame()
        df["dist"] = np.cumsum(hx[selector])  # Poloidal distance
        # df["R"] = self.g["R"][selector[::-1]]
        # df["Z"] = self.g["Z"][selector[::-1]]
        df[param] = self.bal[param][selector]
        
        return df
    
    def find_param(self, name):
        """
        Returns variables that match string
        """
        for param in self.params:
            if name in param: print(param)
        
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