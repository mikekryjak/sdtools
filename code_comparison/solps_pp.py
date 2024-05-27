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
from mpl_toolkits.axes_grid1 import make_axes_locatable

onedrive_path = onedrive_path = str(os.getcwd()).split("OneDrive")[0] + "OneDrive"
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\gridtools"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\sdtools"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\soledge"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages"))


# from gridtools.hypnotoad_tools import *
# from gridtools.b2_tools import *
# from gridtools.utils import *

from hermes3.case_db import *
from hermes3.load import *
from hermes3.named_selections import *
from hermes3.plotting import *
from hermes3.grid_fields import *
from hermes3.accessors import *
from hermes3.utils import *
# from code_comparison.viewer_2d import *
# from code_comparison.code_comparison import *

# import gridtools.solps_python_scripts.setup
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
        
        NOTES ON GUARD CELLS IN SOLPS
        - Total cell count is nx+2, ny+2. Guard cells are there for upper targets but somehow don't get counted (?!). 
          All of the balance file results are with guard cells and this function plots them by default.
          Their position/mask can be found in resignore available in b2fgmtry. 
        - Guard cells are always really tiny!!!
        """
        
        raw_balance = nc.Dataset(os.path.join(path, "balance.nc"))
        
        ## Need to transpose to get over the backwards convention compared to MATLAB
        bal = {}
        for key in raw_balance.variables:
            bal[key] = raw_balance[key][:].transpose()
            
        self.bal = bal
        
        raw_balance.close()

        self.params = list(bal.keys())
        self.params.sort()
        
        # Set up geometry
        
        g = self.g = {}
        
        # Get cell centre coordinates
        R = self.g["R"] = np.mean(bal["crx"], axis=2)
        Z = self.g["Z"] = np.mean(bal["cry"], axis=2)
        self.g["hx"] = bal["hx"]
        self.g["hy"] = bal["hy"]
        self.g["crx"] = bal["crx"]
        self.g["cry"] = bal["cry"]
        self.g["nx"] = self.g["crx"].shape[0]
        self.g["ny"] = self.g["crx"].shape[1]
        
        self.g["Btot"] = bal["bb"][:,:,3]
        self.g["Bpol"] = bal["bb"][:,:,0]
        self.g["Btor"] = np.sqrt(self.g["Btot"]**2 - self.g["Bpol"]**2) 
        
        
        
        leftix = bal["leftix"]+1
        leftix_diff = np.diff(leftix[:,1])
        g["xcut"] = xcut = np.argwhere(leftix_diff<0).squeeze()
        g["leftcut"] = [xcut[0]-2, xcut[1]+2]
        g["rightcut"] = [xcut[4]-1, xcut[3]-1]
        
        omp = self.g["omp"] = int((g["rightcut"][0] + g["rightcut"][1])/2) + 1
        imp = self.g["imp"] = int((g["leftcut"][0] + g["leftcut"][1])/2)
        upper_break = self.g["upper_break"] = g["xcut"][2] + 1
        sep = self.g["sep"] = bal["jsep"][0] + 2
        self.g["xpoints"] = self.find_xpoints()
        
        
        ## Prepare selectors
        
        s = self.s = {}
        s["imp"] = (imp, slice(None,None))
        s["omp"] = (omp, slice(None,None))
        s["outer_lower_target"] = (-2, slice(None,None))
        s["inner_lower_target"] = (1, slice(None,None))
        
        
        # First SOL rings
        for name in ["outer", "outer_lower", "outer_upper", "inner", "inner_lower", "inner_upper"]:
            s[name] = self.make_custom_sol_ring(name, i = 0)
            
        # BOUT++ style regions - needs implementing leftcut, rightcut
        # s["lower_inner_pfr"] = (slice(None, leftcut[0]+2), slice(None, sep+2))
        # s["lower_inner_SOL"] = (slice(None, leftcut[0]+2), slice(sep+2, None))
        # s["inner_core"] = (slice(leftcut[0]+2, leftcut[1]+2), slice(None, sep+2))
        # s["inner_SOL"] = (slice(leftcut[0]+2, leftcut[1]+2), slice(sep+2, None))
        # s["upper_inner_PFR"] = (slice(leftcut[1]+2, upper_break), slice(None, sep+2))
        # s["upper_inner_SOL"] = (slice(leftcut[1]+2, upper_break), slice(sep+2, None))
        # s["upper_outer_PFR"] = (slice(upper_break, rightcut[1]+2), slice(None, sep+2))
        # s["upper_outer_SOL"] = (slice(upper_break, rightcut[1]+2), slice(sep+2, None))
        # s["outer_core"] = (slice(rightcut[1]+2, rightcut[0]+2), slice(None, sep+2))
        # s["outer_SOL"] = (slice(rightcut[1]+2, rightcut[0]+2), slice(sep+2, None))
        # s["lower_outer_PFR"] = (slice(rightcut[0]+2, None), slice(None, sep+2))
        # s["lower_outer_SOL"] = (slice(rightcut[0]+2, None), slice(sep+2, None))
            
        # ## Csalculate array of radial distance from separatrix
        # dist = np.cumsum(self.g["hy"][self.s["omp"]])   # hy is radial length
        # dist = dist - dist[self.g["sep"] - 1]
        # self.g["radial_dist"] = dist
        
    
    def derive_data(self):
        """
        Add new derived variables to the balance file
        This includes duplicates which are in Hermes-3 convention
        """
        
        bal = self.bal
        
        ## EIRENE derived variables have extra values in the arrays
        ## Either 5 or 2 extra zeros at the end in poloidal depending on topology
        double_null = True
        if double_null is True:
            ignore_idx = 5
        else:
            ignore_idx = 2
        
        bal["Td+"] = bal["ti"] / constants("q_e")
        bal["Te"] = bal["te"] / constants("q_e")
        bal["Ne"] = bal["ne"]
        bal["Pe"] = bal["ne"] * bal["te"] * constants("q_e")
        bal["Pd+"] = bal["ne"] * bal["ti"] * constants("q_e")

        bal["Na"] = bal["dab2"][:-ignore_idx, :]  
        bal["Nm"] = bal["dmb2"][:-ignore_idx, :] 
        bal["Nn"] = bal["Na"] + bal["Nm"] * 2
        bal["Ta"] = bal["tab2"][:-ignore_idx, :] / constants("q_e")
        bal["Tm"] = bal["tmb2"][:-ignore_idx, :] / constants("q_e")
        bal["Pa"] = bal["Na"] * bal["Ta"] * constants("q_e")
        bal["Pm"] = bal["Nm"] * bal["Tm"] * constants("q_e")

        bal["Pn"] = bal["Pa"] + bal["Pm"]
        bal["Tn"] = bal["Pn"] / bal["Nn"] / constants("q_e")
        
        self.bal = bal


        
    def make_custom_sol_ring(self, name, i = None, sep_dist = None):
        """
        Inputs
        ------
            name, str:
                "outer", "outer_lower", "outer_upper", "inner", "inner_lower", "inner_upper"
            i, int: 
                SOL ring index (0 = first inside SOL) 
            sep_dist, int
                Separatrix distance in [m]
                
        Returns:
        ------
            selector tuple as (X,Y)
            
        INCLUDES GUARD CELLS
        """
        
        if i != None and sep_dist == None:
            yid = self.g["sep"] + i
        if sep_dist != None and i == None:
            yid = np.argmin(np.abs(self.g["radial_dist"] - sep_dist))
            if yid < self.g["sep"]:
                yid = self.g["sep"]
                print("SOL ring would have been inside the core, forcing to sep")
        if i != None and sep_dist != None:
            raise ValueError("Use i or sep_dist but not both")
        if i == None and sep_dist == None:
            raise ValueError("Provide either i or sep_dist")
        
        selections = {}
        selections["outer"] =       (slice(self.g["upper_break"],None), yid)
        selections["outer_lower"] = (slice(self.g["omp"]+1,None), yid)
        selections["outer_upper"] = (slice(self.g["upper_break"], self.g["omp"]+1), yid)
        selections["inner"] =       (slice(1, self.g["upper_break"]), yid)
        selections["inner_lower"] = (slice(1, self.g["imp"]), yid)
        selections["inner_upper"] = (slice(self.g["imp"], self.g["upper_break"]), yid)
        
        return selections[name]
    

        
    def close(self):
        self.bal.close()
        
    def plot_2d(self, param,
             ax = None,
             norm = None,
             data = np.array([]),
             cmap = "Spectral_r",
             custom_cmap = None,
             antialias = True,
             linecolor = "k",
             linewidth = 0,
             vmin = None,
             vmax = None,
             logscale = False,
             alpha = 1,
             separatrix = True,
             separatrix_kwargs = {},
             grid_only = False,
             cbar = True,
             axis_labels = True):
        
        if custom_cmap != None:
            cmap = custom_cmap
        
        if grid_only is True:
            data = np.zeros_like(self.bal["te"])
        elif len(data)==0:
            data = self.bal[param]
        
        if vmin == None:
            vmin = data.min()
        if vmax == None:
            vmax = data.max()
        if norm == None:
            norm = create_norm(logscale, norm, vmin, vmax)
        if ax == None:
            fig, ax = plt.subplots(dpi = 150)
        else:
            fig = ax.get_figure()
        

        # Following SOLPS convention: X poloidal, Y radial
        crx = self.bal["crx"]
        cry = self.bal["cry"]
        nx, ny = self.g["nx"], self.g["ny"]
        
        
        # print(f"Data shape: {data.shape}")
        # print(f"Grid shape: {nx, ny}")
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
                    np.concatenate([crx[i,j,:][tuple(idx)], cry[i,j,:][tuple(idx)]]).reshape(2,5).T,
                    
                    fill=False,
                    closed=True,
                )
                patches.append(p)
                
        # Polygon colors
        colors = data.flatten()
        
        if grid_only is True:
            polys = mpl.collections.PatchCollection(
                patches, alpha = alpha, norm = norm, cmap = cmap, 
                antialiaseds = antialias,
                edgecolors = linecolor,
                facecolor = "white",
                linewidths = linewidth,
                joinstyle = "bevel")
        
        else:
            polys = mpl.collections.PatchCollection(
                patches, alpha = alpha, norm = norm, cmap = cmap, 
                antialiaseds = antialias,
                edgecolors = linecolor,
                linewidths = linewidth,
                joinstyle = "bevel")
            polys.set_array(colors)
        
        ## Cbar
        if fig != None and grid_only == False and cbar == True:
            # From https://joseph-long.com/writing/colorbars/
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(polys, cax = cax)
            cax.grid(visible = False)
        ax.add_collection(polys)
        ax.set_aspect("equal")
        
        ## Somehow autoscale breaks sometimes
        xmin, xmax = crx.min(), crx.max()
        ymin, ymax = cry.min(), cry.max()
        xspan = xmax - xmin
        yspan = ymax - ymin

        ax.set_xlim(xmin - xspan*0.05, xmax + xspan*0.05)
        ax.set_ylim(ymin - yspan*0.05, ymax + yspan*0.05)
        
        if axis_labels is True:
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
        if grid_only is False:
            ax.set_title(param)
        
        if separatrix is True:
            self.plot_separatrix(ax = ax, **separatrix_kwargs)
            
    def plot_separatrix(self, ax, **separatrix_kwargs):
        
        kwargs = {**{"c" : "white", "ls" : "-"}, **separatrix_kwargs}

        R = self.g["crx"][:,:,0]
        Z = self.g["cry"][:,:,0]
        ax.plot(R[self.s["inner"]], Z[self.s["inner"]], **kwargs)
        ax.plot(R[self.s["outer"]], Z[self.s["outer"]], **kwargs)
        
    def get_1d_radial_data(
        self,
        params,
        region = "omp",
        verbose = False
    ):
        """
        Returns OMP or IMP data from the balance file
        Note that the balance file shape is a transpose of the geometry shape (e.g. crx)
        and contains guard cells.        
        """

        if type(params) == str:
            params = [params]
        
        if any([region in name for name in ["omp", "imp"]]):
            p = self.s[region] 
        else:
            raise Exception(f"Unrecognised region: {region}")
    
        df = pd.DataFrame()
        df["dist"] = self.g["R"][p[0], p[1]] - self.g["R"][p[0], self.g["sep"]] 
        df["R"] = self.g["R"][p[0], p[1]]
        df["Z"] = self.g["Z"][p[0], p[1]]

        for param in params:
            df[param] = self.bal[param][p[0], :]
        
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
            
    def find_xpoints(self, plot = False):
        """
        Returns poloidal IDs of xpoints, works for double null only
        the IDs are for the cell immediately after the X-point
        in the direction of ascending X (poloidal) coordinate.
        They are arranged clockwise starting at inner lower
        """

        x = range(self.g["nx"])
        R = self.g["R"][:,self.g["sep"]]

        inner_R = self.g["R"][slice(None, self.g["upper_break"]),self.g["sep"]]
        inner_x = np.array(range(self.g["upper_break"]))


        outer_R = self.g["R"][slice(self.g["upper_break"], None),self.g["sep"]]
        outer_x = np.array(range(self.g["upper_break"], self.g["nx"]))

        import scipy

        inner_ids = scipy.signal.find_peaks(inner_R, threshold = 0.001)
        outer_ids = scipy.signal.find_peaks(outer_R*-1, threshold = 0.001)

        if (len(inner_ids[0]) != 2) or (len(outer_ids[0]) != 2):
            raise Exception("Issue in peak finder")

        if plot is True:
            fig, axes = plt.subplots(1,2, figsize = (8,4))
            ax = axes[0]
            ax.plot(inner_x, inner_R)
            for i in inner_ids[0]:
                ax.scatter(inner_x[i], inner_R[i])
            ax = axes[1]
            ax.plot(outer_x, outer_R)
            for i in outer_ids[0]:
                ax.scatter(outer_x[i], outer_R[i])

        xpoints = [
            inner_ids[0][0], 
            inner_ids[0][1]+1, 
            outer_ids[0][0]+ self.g["upper_break"] + 1, 
            outer_ids[0][1]+ self.g["upper_break"]
            ]

        return xpoints
    
    def plot_selection(self, sel, ylims = (None, None), xlims = (None, None)):
        """
        Plots scatter of selected points according to the tuple sel
        which contains slices in (X, Y)
        Examples provided in self.s
        """
        
        data = self.bal["ne"][:]
        data = np.zeros_like(data)

        data[sel] = 1

        self.plot_2d("", data = data, cmap = mpl.colors.ListedColormap(["lightgrey", "deeppink"]),
             antialias = True, 
             cbar = False,
             separatrix_kwargs = dict(lw = 0.5, c = "skyblue"))
        
        if xlims != (None, None):
            ax.set_xlim(xlims)
            
        if ylims != (None, None):
            ax.set_ylim(ylims)
        
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

def read_display_tallies(path):
    """ 
    This reads a csv file coming from a copy paste of the output of the display_tallies SOLPS routine
    It parses the table into three dataframes corresponding to the volumetric tallies,
    and the tallies of fluxes between regions in the x and y directions
    Each variable comes in three flavours: suffixless = total, _n = neutrals, _i = ions
    """
    
    # Read display tallies csv copied from console
    df = pd.read_csv(path, skiprows=2, names = range(16), sep = r"\s+", index_col = 0)

    # Make a col "param" with parameters - rest of columns are the 0-14 regions (double null only)
    df = df[df.index.notnull()].reset_index()
    df.columns = df.columns - 1
    df = df.rename({-1 : "param"}, axis = 1)
    
    # Some lines have no parameter title and the delimiter parsing makes it start as a number
    # Shift those back to be in line with the others
    df[df["param"].str.contains("E\+")] = df[df["param"].str.contains("E\+")].shift(axis=1)

    params = df["param"][df["param"].str.contains("D0|D\+|None") == False].values

    for i in df.index:
        # Empty rows due to the shifting above are now labelled correctly
        if df.loc[i, "param"] == None:
            df.loc[i, "param"] = df.loc[i-1, "param"]
            df.loc[i-1, "param"] = None

        # Label D0 as _n, D+1 as _i and D0 + D+1 as the original parameter
        if df.loc[i, "param"] == "D0":
            name = df.loc[i-1, "param"]
            df.loc[i, "param"] = name+"_n"
            df.loc[i+1, "param"] = name+"_i"
            df.loc[i-1] = df.loc[i] + df.loc[i+1]   # The name without suffix becomes sum of neutrals and ions
            df.loc[i-1, "param"] = name
            
    # Remove all None junk
    df = df[~df["param"].isna()]

    for i in range(15):
        df[i] = df[i].astype(float)

    # Parse the volumetric regions
    dfreg = df[((df["param"].str.contains("x|y") == False) & (df["param"].str.contains("reg") == True))]
    dfreg = dfreg.rename({
        0 : "total",
        1 : "inner_core",
        2 : "inner_sol",
        3 : "inner_lower_target",
        4 : "inner_upper_target",
        5 : "outer_core",
        6 : "outer_sol",
        7 : "outer_upper_target",
        8 : "outer_lower_target"
    }, axis = 1
    ).dropna(axis=1)

    # Parse the fluxes between regions in the X direction
    dfxreg = df[((df["param"].str.contains("xreg") == True))]
    dfxreg = dfxreg.rename({
        0 : "total",
        1 : "inner_lower_target",
        2 : "inner_lower_entrance",
        3 : "inner_upper_entrance",
        4 : "inner_upper_target",
        5 : "outer_upper_target",
        6 : "outer_upper_entrance",
        7 : "outer_lower_entrance",
        8 : "outer_lower_target",
        9 : "lower_inner_pfr_to_inner_core",
        10 : "inner_core_to_inner_upper_pfr",
        11 : "outer_upper_pfr_to_outer_core",
        12 : "outer_core_to_outer_lower_pfr"
    }, axis = 1
    ).dropna(axis=1)
    
    # Parse the fluxes between regions in the Y direction
    dfyreg = df[((df["param"].str.contains("yreg") == True))]
    dfyreg = dfyreg.rename({
        0 : "total",
        1 : "inner_lower_pfr",
        2 : "inner_core",
        3 : "inner_upper_pfr",
        4 : "inner_separatrix",
        5 : "inner_lower_sol",
        6 : "inner_sol",
        7 : "inner_upper_sol",
        8 : "outer_upper_pfr",
        9 : "outer_core",
        10 : "outer_lower_pfr",
        11 : "outer_separatrix",
        12 : "outer_upper_sol",
        13 : "outer_sol",
        14 : "outer_lower_sol"
    }, axis = 1
    ).dropna(axis=1)
    
    for dfout in [dfreg, dfxreg, dfyreg]:
        dfout.index = dfout.pop("param")

    return {"reg":dfreg, "xreg":dfxreg, "yreg":dfyreg }