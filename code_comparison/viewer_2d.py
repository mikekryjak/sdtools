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

sys.path.append(r'C:\Users\mikek\OneDrive\Project\python-packages')
sys.path.append(r'C:\Users\mikek\OneDrive\Project\python-packages\soledge')

try:
    import gridtools.solps_python_scripts.setup
    from gridtools.solps_python_scripts.plot_solps       import plot_1d, plot_2d
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
        "Nd" : "pdenn",   # Combined atoms+molecules. Custom made by matteo. Atoms: pdena, Molecules: pdenm
        "Td" : "tdena",    # Compare only atom temperature, ignore molecules (more physical)
        "Sd+_iz" : "AMJUEL_H.4_2.1.5_3"
    }
    
    soledge = {
        "Ne" : "Dense",
        "Te" : "Tempe",
        "Nd+" : "Densi",
        "Td+" : "Tempi",
        "Vd+" : "velocityi",
        "Pd+" : "Ppi",
        "Pe" : "Ppe",
        "Rd+_ex" : "IRadi",
        "Nd" : "Nni",
        "Td" : "Tni",
    }
    
    if code == "solps":
        return solps[x]
    elif code == "soledge":
        return soledge[x]

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
        
        hermes_present = False
        for case in cases.keys():
            if cases[case]["code"] == "hermes":
                hermes_present = True
        self.param = param
        self.cases = cases
        num_cases = len(cases.keys())
        

        # Find ranges if not provided
        if hermes_present is True:
            self.max, self.min = self.find_ranges()
            
            if vmin == None:
                vmin = self.min
            else:
                self.min = vmin
            if vmax == None:
                vmax = self.max
            else:
                self.max = vmax
        else:
            self.min = None
            self.max = None
        
        norm = _create_norm(logscale, None, vmin, vmax)

        fig = plt.figure(dpi=dpi)
        fig.set_figheight(6)
        fig.set_figwidth(num_cases*4)

        # Plot grid
        gs0a = mpl.gridspec.GridSpec(
                                        ncols=num_cases+1, nrows=2,
                                        width_ratios = [1]*num_cases + [0.1], # cbar, empty space, control,
                                        height_ratios = [0.95, 0.05],
                                        wspace = wspace
                                        )
    
        

        axes = [None]*(num_cases+3)


        for i, casename in enumerate(cases.keys()):
            
            # All plots after the first one share x and y axes
            if i == 0:
                axes[i] = fig.add_subplot(gs0a[i])
            else:
                axes[i] = fig.add_subplot(gs0a[i], sharex=axes[0], sharey=axes[0])
            
            model = cases[casename]
            
            # HERMES PLOTTING------------------------------------
            if model["code"] == "hermes":
                print(casename)
                data = model["ds"][param].hermesm.clean_guards()
                data.bout.polygon(ax = axes[i], 
                                    add_colorbar = False, norm = norm,
                                    separatrix = True, cmap = cmap, 
                                    antialias = False,
                                    linewidth = 0,
                                    )
                axes[i].set_title(f"Hermes-3\n{casename}")#\n{param}")
                
            # SOLPS PLOTTING------------------------------------
            elif model["code"] == "solps":
                if "param_override" in model.keys():
                    solps_name = model["param_override"]
                else:
                    try:
                        solps_name = name_parser(param,"solps")
                        
                    except:
                        print(f"Parameter {param} not found in SOLPS")
                        solps_name = None
                    
                solps_cbar = False if hermes_present is True else True
                
                if solps_name != None:
                    plot_2d(fig, axes[i], 
                            where = model["path"], 
                            what = solps_name, 
                            cmap = cmap, 
                            scale = ("log" if logscale is True else "linear"), 
                            vmin = vmin, 
                            vmax = vmax, 
                            plot_cbar = solps_cbar)
                    axes[i].set_title(f"SOLPS\n{casename}\n{solps_name}")
                
                else:
                    axes[i].set_title(f"SOLPS\{casename}: {param} not found")
                    
            # SOLEDGE PLOTTING------------------------------------
            elif model["code"] == "soledge":

                try:
                    soledge_name = name_parser(param, "soledge")
                except:
                    print(f"Parameter {param} not found in SOLEDGE")
                    soledge_name = None
            
                if soledge_name != None:
                    soledge_plot = SOLEDGEplot(path = model["path"], param = soledge_name)
                    soledge_plot.plot(ax = axes[i], norm = norm, cmap = cmap)
                
                axes[i].set_title(f"SOLEDGE2D\n{casename}")#\nParameter") = {soledge_name}")
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
        if hermes_present is True:
            axes[-1] = fig.add_subplot(gs0a[0,-1])   
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = plt.colorbar(mappable = sm, cax = axes[-1], label = param)

        if hermes_present is True:
            slider = RangeSlider(
                fig.add_axes([0.2, 0.05, 0.5, 0.1]), "Colour limits",   # left, bottom, width, height
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
                if cases[list(cases.keys())[i]]["code"] == "solps" and logscale == True:
                    artist.norm.vmin = np.log10(val[0])
                    artist.norm.vmax = np.log10(val[1])
                else:
                    artist.norm.vmin = val[0]
                    artist.norm.vmax = val[1]
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events() # https://stackoverflow.com/questions/64789437/what-is-the-difference-between-figure-show-figure-canvas-draw-and-figure-canva
                
        # slider.on_changed(update)

            
        
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
    
class SOLEDGEplot():
    """
    Reads parameter data from SOLEDGE case
    Finds min and max
    Can plot it on a provided axis. Accepts a custom norm
    Provide it a path to the case directory and parameter name in SOLEDGE convention
    Parameter names are available in .Triangles.Vnames
    """
    def __init__(self, path, param):
        
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
        iPar = Plasmas[species_idx][0].Triangles.VNames.index(param)	
        self.plot_data = Plasmas[1][0].Triangles.Values[iPar]
        self.vmin = min(self.plot_data)
        self.vmax = max(self.plot_data)
        
        # Return to reasonable units
        # if "Temp" in param:
        #     self.plot_data *= 1e-3
        
        # Extract geom and make triangulation
        if_tri	 = h5py.File(os.path.join(path,"triangles.h5"), "r")
        TriKnots = h5_read(if_tri,"triangles/tri_knots")
        TriKnots = TriKnots - 1 										#Matlab/Fortan to python indexes
        self.R		 = h5_read(if_tri,"knots/R")*0.01
        self.Z		 = h5_read(if_tri,"knots/Z")*0.01
        if_tri.close()
        self.TripTriang = tri.Triangulation(self.R, self.Z, triangles=TriKnots)
        
        # Extract mesh for eqb 
        self.Config = load_soledge_mesh_file(os.path.join(path,"mesh.h5"))
        
        
        
    def plot(self, ax, norm = None, sep = True, cmap = "Spectral_r"):
        ax.tripcolor(self.TripTriang, self.plot_data, norm = norm, cmap = cmap,  linewidth=0)
        ax.set_aspect("equal")
        if sep is True:
            
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
    
    
def _create_norm(logscale, norm, vmin, vmax):
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
            norm = mpl.colors.SymLogNorm(linear_threshold, vmin=vmin, vmax=vmax)
    elif norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return norm