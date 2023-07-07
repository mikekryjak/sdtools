import numpy as np
import os, sys
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri						as tri
import xbout
from matplotlib.widgets import RangeSlider, TextBox
from .code_comparison import parse_solps
import h5py
import netCDF4 as nc

onedrive_path = onedrive_path = str(os.getcwd()).split("OneDrive")[0] + "OneDrive"
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\sdtools"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\soledge"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages"))

from hermes3.utils import *
try:
    import gridtools.solps_python_scripts.setup
    from gridtools.solps_python_scripts.plot_solps       import plot_1d, plot_2d
    from gridtools.solps_python_scripts.read_b2fgmtry import *
except:
    print("Gridtools not found")
    
# SOLEDGE functions
from files.load_plasma_files						import load_plasma_files
from files.load_soledge_mesh_file				import load_soledge_mesh_file
from routines.h5_routines							import h5_read

def name_parser(x, code):
    
    solps = {
        "Ne" : "ne",
        "Te" : "te",
        "Td+" : "ti",   
        "Nd" : "pdena",   # Combined atoms+molecules. Custom made by matteo. Atoms: pdena, Molecules: pdenm
        "Td" : "tdena",    # Compare only atom temperature, ignore molecules (more physical)
        "Sd+_iz" : "AMJUEL_H.4_2.1.5_3",
        # "R" : "b2ra"
    }
    
    soledge = {
        "Ne" : "Dense",
        "Te" : "Tempe",
        "Nd+" : "Densi",
        "Td+" : "Tempi",
        "Vd+" : "velocityi",
        "Pd+" : "Ppi",
        "Pe" : "Ppe",
        "Rd+_ex" : "IRadi",   # Assumes only ions 
        "Rtot" : "TotRadi",
        "Nd" : "Nni",
        "Td" : "Tni",
    }
    
    if code == "solps":
        if x in solps:
            return solps[x]
        else:
            raise Exception(f"Unknown variable: {x}")
        
    elif code == "soledge":
        if x in soledge:
            return soledge[x]
        else:
            raise Exception(f"Unknown variable: {x}")
        
    else:
        raise Exception(f"Unknown code: {code}")

class viewer_2d():
    """
    Pass a case:
    case = {
        <case_name> : {"name" : <case name>, "code" : "hermes", "ds" : <hermes-3 dataset>},
        <case_name> : {"name" : <case name>, "code" : "solps", "path" : <path to SOLPS dir>}
        <case_name> : {"code" : "soledge", "path": <path to SOLEDGE dir>}
        }
    """
    def __init__(self,
                 param,
                 cases,
                 vmin = None,
                 vmax = None,
                 xlim = (None,None),
                 ylim = (None,None),
                 logscale = True,
                 dpi = 120,
                 wspace = 0.05,
                 cmap = "Spectral_r"):
        
        plots = []
        for case in cases:
            if case["code"] == "hermes":
                plots.append(HermesPlot(case["ds"], param = param))
            
            elif case["code"] == "solps":
                plots.append(SOLPSplot(case["path"], param = param))
                
            elif case["code"] == "soledge":
                plots.append(SOLEDGEplot(case["path"], param = param))
        self.cases = cases
        self.plots = plots
        num_cases = len(cases)
        

        # Find ranges if not provided
        vlims = {"min":[], "max":[]}
        Rlims = {"min":[], "max":[]}
        Zlims = {"min":[], "max":[]}

        for plot in plots:
            vlims["min"].append(plot.vmin)
            vlims["max"].append(plot.vmax)
            Rlims["min"].append(plot.Rlim[0]); Rlims["max"].append(plot.Rlim[1])
            Zlims["min"].append(plot.Zlim[0]); Zlims["max"].append(plot.Zlim[1])
            
       
        # Get color limits
        self.min = min(vlims["min"]) if vmin == None else vmin
        self.max = max(vlims["max"]) if vmax == None else vmax
        norm = create_norm(logscale, None, self.min, self.max)
        
        # Get plot size
        Rmin = min(Rlims["min"]); Rmax = max(Rlims["max"])
        Zmin = min(Zlims["min"]); Zmax = max(Zlims["max"])
        box_width = Rmax - Rmin
        box_height = Zmax - Zmin
        box_aspect_ratio = box_height / box_width

        fig = plt.figure(dpi=dpi)
        scale = 3
        fig.set_figwidth(1 * scale * num_cases)
        fig.set_figheight(1 * scale * box_aspect_ratio)

        # Plot grid
        gs0a = mpl.gridspec.GridSpec(
                                        ncols=num_cases, nrows=2,
                                        width_ratios = [1]*num_cases,
                                        height_ratios = [0.90, 0.1],
                                        wspace = wspace
                                        )

        axes = [None]*(num_cases)

        for i, plot in enumerate(plots):
            
            # All plots after the first one share x and y axes
            if i == 0:
                axes[i] = fig.add_subplot(gs0a[i])
            else:
                axes[i] = fig.add_subplot(gs0a[i], sharex=axes[0], sharey=axes[0])
            
            case = cases[i]
            plot.plot(ax = axes[i], norm = norm, cmap = cmap, separatrix = True)
            axes[i].set_title(f"{case['code']}: {case['name']}")
            
            # SET LIMITS------------------------------------
            if xlim != (None, None):
                axes[i].set_xlim(xlim)
            else:
                axes[i].set_xlim(0.15, 0.78)
                
            if ylim != (None, None):
                axes[i].set_ylim(ylim)
            else:
                axes[i].set_ylim(-0.88,0.1)
            
            # Take out Y markings from plots after first one
            if i != 0:
                axes[i].set_ylabel("")
                axes[i].set_xlabel("R [m]")
                axes[i].tick_params(axis="y", which="both", left=False,labelleft=False)
                
        # Add colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cax = fig.add_axes([
            axes[-1].get_position().x1+0.01,
            axes[-1].get_position().y0,0.02,
            axes[-1].get_position().height])
        plt.colorbar(mappable = sm, cax=cax, label = param) # Similar to fig.colorbar(im, cax = cax)

        # Add slider
        slider = RangeSlider(
                fig.add_axes([0.3, 0.12, 0.5, 0.1]), "Colour limits",   # left, bottom, width, height
                self.min, self.max,
                orientation = "horizontal",
                valinit = (self.min, self.max)
                )
            

        artists = []
        
        # This is to account for missing artists if one case has no param
        for i in range(num_cases):
            try:
                artists.append(axes[i].collections[0])
            except:
                pass
        # artists = artists.append(cbar)

        def update(val):
            slider.ax.set_ylim(self.min, self.max) # This is inexplicably needed otherwise it freezes
            
            cbar.norm.vmin = val[0]
            cbar.norm.vmax = val[1]
                
            for i, artist in enumerate(artists):
                if logscale == True:
                    artist.norm.vmin = np.log10(val[0])
                    artist.norm.vmax = np.log10(val[1])
                else:
                    artist.norm.vmin = val[0]
                    artist.norm.vmax = val[1]
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events() # https://stackoverflow.com/questions/64789437/what-is-the-difference-between-figure-show-figure-canvas-draw-and-figure-canva
                
        slider.on_changed(update)

            
        
    def find_ranges(self):
        """
        Find min and max for plotting.
        Only uses hermes so far
        """
        max = []
        min = []
        
        for casename in self.cases.keys():
            case = self.cases[casename]
            if case["code"] == "hermes":
                max.append(case["ds"][self.param].values.max())
                min.append(case["ds"][self.param].values.min())
                
        return np.max(max), np.min(min)
    
class HermesPlot():
    """
    Wrapper around Hermes to provide identical functionality to SOLEDGE and SOLPS plots
    """
    def __init__(self, ds, param, clean_guards = True):
        self.dataarray = ds[param]
        if clean_guards is True:
            self.dataarray = self.dataarray.hermesm.clean_guards()
            
        self.data = self.dataarray.values
        self.vmin = np.nanmin(self.data)
        self.vmax = np.nanmax(self.data)
        self.Rlim = [ds["R"].values.min(), ds["R"].values.max()]
        self.Zlim = [ds["Z"].values.min(), ds["Z"].values.max()]
        
    def plot(self, ax, **kwargs):
        self.dataarray.bout.polygon(ax = ax, add_colorbar = False, **kwargs)
        
        
class SOLEDGEplot():
    """
    Reads parameter data from SOLEDGE case
    Finds min and max
    Can plot it on a provided axis. Accepts a custom norm
    Provide it a path to the case directory and parameter name in SOLEDGE convention
    Parameter names are available in .Triangles.Vnames
    """
    def __init__(self, path, param):
        
        soledgeparam = name_parser(param, "soledge")
        
        with HiddenPrints():
            Plasmas = load_plasma_files(path, nZones=0, Evolution=0, iPlasmas=[0,1])
        if param.endswith("e"):
            # Electrons are index 0
            species_idx = 0
        elif param.endswith("i"):
            # Ions and neutrals are index 1
            species_idx = 1
        else:
            raise Exception("Parameter name must end in e or i")

        # Extract parameter data and find ranges
        iPar = Plasmas[species_idx][0].Triangles.VNames.index(soledgeparam)	
        self.data = Plasmas[1][0].Triangles.Values[iPar]
        self.vmin = min(self.data)
        self.vmax = max(self.data)
        
        
        # Return to reasonable units
        # if "Temp" in param:
        #     self.plot_data *= 1e-3
        
        # Extract geom and make triangulation
        with HiddenPrints():
            if_tri	 = h5py.File(os.path.join(path,"triangles.h5"), "r")
            TriKnots = h5_read(if_tri,"triangles/tri_knots", messages = 0)
            TriKnots = TriKnots - 1 										#Matlab/Fortan to python indexes
            self.R		 = h5_read(if_tri,"knots/R", messages = 0)*0.01
            self.Z		 = h5_read(if_tri,"knots/Z", messages = 0)*0.01
            if_tri.close()
            self.TripTriang = tri.Triangulation(self.R, self.Z, triangles=TriKnots)
        
            # Extract mesh for eqb 
            self.Config = load_soledge_mesh_file(os.path.join(path,"mesh.h5"))
            
        self.Rlim = [self.R.min(), self.R.max()]
        self.Zlim = [self.Z.min(), self.Z.max()]
        
        
        
    def plot(self, ax, 
             norm = None,
             vmin = None, 
             vmax = None, 
             logscale = True, 
             separatrix = True, 
             cmap = "Spectral_r"):
        
        self.vmin = min(self.data) if vmin is None else vmin
        self.vmax = max(self.data) if vmax is None else vmax
            
        if norm is None:
            norm = _create_norm(logscale, norm, self.vmin, self.vmax)
            
        
        ax.tripcolor(self.TripTriang, self.data, norm = norm, cmap = cmap,  linewidth=0)
        ax.set_aspect("equal")
        if separatrix is True:
            
            lw = 2
            c = "w"
            lhs, rhs = self._get_rz_sep()
            ax.plot(lhs["R"], lhs["Z"], lw = lw, c = c)
            ax.plot(rhs["R"], rhs["Z"], lw = lw, c = c)
        # ax.tripcolor(self.R, self.Z, self.plot_data, norm = norm, cmap = cmap,antialiaseds = True, linewidth=0)
    
    def _get_rz_sep(self):

        """
        Queries grid regions to obtain separatrix R and Z coordinates
        Extends to the equilibium boundary
        Requires a Config (grid/eqb) file with MagZones computed
        Returns lhs and rhs for each of the sides of the separatrix for easy contiguous line plots
        Those are dicts with keys R and Z for the respective coords
        """
        Config = self.Config
        sep = dict()
        sep["UOT"] = Config.MagZones[2].north
        sep["UIT"] = Config.MagZones[3].north
        sep["LIT"] = Config.MagZones[4].north
        sep["LOT"] = Config.MagZones[5].north

        sep["ISEP"] = Config.MagZones[0].north
        sep["OSEP"] = Config.MagZones[1].north

        lhs = dict()
        rhs = dict()
        for coord in ["R", "Z"]:
            # lhs[coord] = np.
            lhs[coord] = sep["LIT"].__dict__[coord]
            lhs[coord] = np.concatenate([lhs[coord], sep["ISEP"].__dict__[coord]])
            lhs[coord] = np.concatenate([lhs[coord], sep["UIT"].__dict__[coord]])
            rhs[coord] = sep["UOT"].__dict__[coord]
            rhs[coord] = np.concatenate([rhs[coord], sep["OSEP"].__dict__[coord]])
            rhs[coord] = np.concatenate([rhs[coord], sep["LOT"].__dict__[coord]])
            
        return lhs, rhs
    
class SOLPSplot():
    """ 
    Wrapper for plotting SOLPS data from a balance file
    """
    def __init__(self, path, data = None, param = None):
           
        solpsparam = name_parser(param, "solps")
        bal = nc.Dataset(os.path.join(path, "balance.nc"))
        if param != None:
            self.data = bal[solpsparam][:]
            if any([x in param for x in ["Te", "Td+", "Td"]]):
                self.data /= constants("q_e")   # Convert K -> eV
        elif data != None:
            self.data = data
        else:
            raise Exception("Provide either the data or the parameter name")

        g = read_b2fgmtry(where=path)
        self.g = g
        
        crx = bal["crx"]
        cry = bal["cry"]

        # Following SOLPS convention: X poloidal, Y radial
        Nx = crx.shape[2]
        Ny = crx.shape[1]
        
        
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
        self.patches = patches
        # Get data
        self.vmin = self.data.min()
        self.vmax = self.data.max()
        self.Rlim = [crx[0,:,:].max(), crx[0,:,:].min()]
        self.Zlim = [cry[0,:,:].max(), cry[0,:,:].min()]
        self.variables = bal.variables
        

    def plot(self, 
             ax = None,
             fig = None,
             norm = None, 
             cmap = "Spectral_r",
             antialias = False,
             linecolor = "k",
             linewidth = 0,
             vmin = None,
             vmax = None,
             logscale = False,
             alpha = 1,
             separatrix = True):
        
        if vmin == None:
            vmin = self.vmin
        if vmax == None:
            vmax = self.vmax
        if norm == None:
            norm = xbout.plotting.utils._create_norm(logscale, norm, vmin, vmax)
        if ax == None:
            fig, ax = plt.subplots(dpi = 150)
            

        logscale = False


        cmap = "Spectral_r"

        # Polygon colors
        
        colors = self.data.transpose().flatten()
        polys = mpl.collections.PatchCollection(
            self.patches, alpha = alpha, norm = norm, cmap = cmap, 
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
        






class viewer_2d_next():
    """
    Pass a case:
    case = {
        <case_name> : {"name" : <case name>, "code" : "hermes", "ds" : <hermes-3 dataset>},
        <case_name> : {"name" : <case name>, "code" : "solps", "path" : <path to SOLPS dir>}
        <case_name> : {"code" : "soledge", "path": <path to SOLEDGE dir>}
        }
    """
    def __init__(self,
                 param,
                 cases,
                 vmin = None,
                 vmax = None,
                 logscale = True,
                 xlim = (None,None),
                 ylim = (None,None),
                 dpi = 120,
                 cmap = "Spectral_r"):
        

        self.param = param
        self.cases = cases
        num_cases = len(cases.keys())
        self.max, self.min = self.find_ranges()

        # Find ranges if not provided
        if vmin == None:
            vmin = self.min
        if vmax == None:
            vmax = self.max

        fig = plt.figure(dpi=120)
        fig.set_figheight(6)
        fig.set_figwidth(num_cases*3.5)

        gs0 = mpl.gridspec.GridSpec(ncols=1, nrows=3, 
                                    height_ratios = [0.9, 0.05, 0.05],
                                    hspace = 0.4,
                                    left = 0.1,
                                    right = 0.9,
                                    top = 0.9,
                                    bottom = 0)
        # Plot grid
        gs0a = mpl.gridspec.GridSpecFromSubplotSpec(
                                        subplot_spec=gs0[0],
                                        ncols=num_cases+1, nrows=1,
                                        width_ratios = [1]*num_cases + [0.1], # cbar, empty space, control
                                        )
        
        # Widget grid
        gs0b = mpl.gridspec.GridSpecFromSubplotSpec(
                                        subplot_spec=gs0[1],
                                        ncols=3, nrows=1,
                                        width_ratios = [0.2, 0.2, 0.6], # box, box, slider
                                        wspace = 0.5,
                                        )
        
        # Dummy gridspec for space
        gs0c = mpl.gridspec.GridSpecFromSubplotSpec(
                                        subplot_spec=gs0[2],
                                        ncols=1, nrows=1,
                                        )
        

        axes = [None]*(num_cases+3)


        for i, casename in enumerate(cases.keys()):
            
            # All plots after the first one share x and y axes
            if i == 0:
                axes[i] = fig.add_subplot(gs0a[i])
            else:
                axes[i] = fig.add_subplot(gs0a[i], sharex=axes[0], sharey=axes[0])
            
            model = cases[casename]
            
            if model["code"] == "hermes":
                model["ds"][param].bout.polygon(ax = axes[i], add_colorbar = False, logscale = logscale, separatrix = True, cmap = cmap, vmin = vmin, vmax = vmax, antialias = False)
                axes[i].set_title(f"Hermes-3\{casename}")
                
            elif model["code"] == "solps":
                try:
                    solps_name = name_parser(param,"solps")
                    
                except:
                    print(f"Parameter {param} not found in SOLPS")
                    solps_name = None
                    
                if solps_name != None:
                    plot_2d(fig, axes[i], where = model["path"], what = name_parser(param,"solps"), cmap = "Spectral_r", scale = ("log" if logscale is True else "linear"), vmin = vmin, vmax = vmax, plot_cbar = False)
                    axes[i].set_title(f"SOLPS\{casename}")
                
                else:
                    axes[i].set_title(f"SOLPS\{casename}: {param} not found")
                
            if xlim != (None, None):
                axes[i].set_xlim(xlim)
            else:
                axes[i].set_xlim(0.15, 0.78)
                
            if ylim != (None, None):
                axes[i].set_ylim(ylim)
            else:
                axes[i].set_ylim(-0.88,0.1)

            
            # Take out Y markings from plots after first one
            if i != 0:
                axes[i].set_ylabel("")
                axes[i].set_xlabel("R [m]")
                axes[i].tick_params(axis="y", which="both", left=False,labelleft=False)
                
        # Add colorbar
        axes[-1] = fig.add_subplot(gs0a[-1])   
        norm = xbout.plotting.utils._create_norm(logscale, None, vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(mappable = sm, cax = axes[-1], label = param)

        
        boxmin = TextBox(
            ax = fig.add_subplot(gs0b[0]), 
            label = "Min", 
            initial = f"{self.min:.2e}",
            textalignment="center")
        
        boxmax = TextBox(
            ax = fig.add_subplot(gs0b[1]), 
            label = "Max", 
            initial = f"{self.max:.2e}",
            textalignment="center")
        
        slider = RangeSlider(
            fig.add_subplot(gs0b[2]), "Colour limits",
            self.min, self.max,
            orientation = "horizontal",
            valinit = (self.min, self.max)
            )

        artists = [axes[i].collections[0] for i in range(num_cases)]
        # artists = artists.append(cbar)
        
        def set_vmin(val):
            cbar.norm.vmin = val
                
            for i, artist in enumerate(artists):
                if cases[list(cases.keys())[i]]["code"] == "solps" and logscale == True:
                    artist.norm.vmin = np.log10(val)
                else:
                    artist.norm.vmin = val
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events() # https://stackoverflow.com/questions/64789437/what-is-the-difference-between-figure-show-figure-canvas-draw-and-figure-canva

        def update(val):
            slider.ax.set_ylim(self.min, self.max) # This is inexplicably needed otherwise it freezes
            
            cbar.norm.vmin = val[0]
            cbar.norm.vmax = val[1]
                
            for i, artist in enumerate(artists):
                if cases[list(cases.keys())[i]]["code"] == "solps" and logscale == True:
                    artist.norm.vmin = np.log10(val[0])
                    artist.norm.vmax = np.log10(val[1])
                else:
                    artist.norm.vmin = val[0]
                    artist.norm.vmax = val[1]
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events() # https://stackoverflow.com/questions/64789437/what-is-the-difference-between-figure-show-figure-canvas-draw-and-figure-canva
                
        slider.on_changed(update)
        boxmin.on_text_change(set_vmin)
            
        
    def find_ranges(self):
        """
        Find min and max for plotting.
        Only uses hermes so far
        """
        max = []
        min = []
        
        for casename in self.cases.keys():
            case = self.cases[casename]
            if case["code"] == "hermes":
                max.append(case["ds"][self.param].values.max())
                min.append(case["ds"][self.param].values.min())
                
        return np.max(max), np.min(min)
    
    
