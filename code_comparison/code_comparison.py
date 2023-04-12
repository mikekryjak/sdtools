import numpy as np
import os, sys
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import xbout
from matplotlib.widgets import RangeSlider, TextBox


import re
from collections import defaultdict

sys.path.append(r'C:\Users\mikek\OneDrive\Project\python-packages')

try:
    import gridtools.solps_python_scripts.setup
    from gridtools.solps_python_scripts.plot_solps       import plot_1d, plot_2d
except:
    print("Gridtools not found")



from hermes3.utils import *

def save_last10s_subset(solps_path, destination_path):
    solps = file_read(solps_path)

    dfs = dict()
    for key in solps.keys():
        if any([x in key for x in ["ti3", "te3", "ne3", "dab23", "dmb23"]]):
            dfs[key] = pd.DataFrame(solps[key])

    file_write(dfs, destination_path)

class SOLEDGEdata:

    def __init__(self):
        self.regions = dict()
        self.code = "SOLEDGE2D"
    
    def read_csv(self, path, mode):
        
        df = pd.read_csv(path)
        
        
        if mode == "plot1d_omp":
            
            self.process_plot1d(df, "omp")
            
        elif mode == "plot1d_imp":
            
            self.process_plot1d(df, "imp")
            
        if mode == "wall_ntmpi":
            self.wallring = df
            self.process_ring()

            
    def process_plot1d(self, df, name):
        """ 
        Process the csv file named "plot1d" which has radial profiles at the OMP.
        """
        self.regions[name] = df
        self.regions[name] = self.regions[name].rename(columns = {'Dense' : "Ne", 
                                            'Tempe':"Te", 
                                            'Densi':"Nd+", 
                                            'Tempi':"Td+",
                                            'velocityi':"Vd+",
                                            'Ppi':"Pd+",
                                            'IRadi':"Rd+_ex",
                                            "DIST":"x"})
        self.regions[name] = self.regions[name].set_index("x")
        self.regions[name].index.name = "pos"
        
    def process_ring(self):
        """ 
        Process the csv file named "wall_ntmpi" which is the
        final cell ring going all the way around the domain counter-clockwise.
        """
        
        self.wallring = self.wallring.rename(columns = {
                            ' ne (*10^19 m-3)' : "Ne", 
                            ' Te (eV)':"Te", 
                            ' Jsat_e (kA/m^2)':"Jsat_e", 
                            ' Mach_e': "Me", 
                            ' ni (*10^19 m-3)': "Nd+", 
                            ' Ti (eV)' : "Td+",
                            ' Jsat_i (kA/m^2)': "Jsat_d+", 
                            ' Mach_i': "Md+,",
                            ' Ioniz_H' : "Ioniz_H",
                            'l (m)' : "l"
                            })
        
        self.wallring["Ne"] *= 10**19
        self.wallring["Nd+"] *= 10**19
        self.wallring = self.wallring.set_index("l")
        self.wallring.index.name = "pos"
        
        # Find peaks in temperature which correspond to the four divertors
        peaks = scipy.signal.find_peaks(self.wallring["Te"], prominence=10)[0]

        strikepoints = dict()
        
        # Create copy of each df for each target with 0 being the strikepoint.
        for i, target in enumerate(["inner_lower", "outer_lower", "outer_upper", "inner_upper"]):
            strikepoints[target] = self.wallring.index[peaks[i]]
            df = self.wallring.copy()
            df.index -= strikepoints[target]
            
            self.regions[target] = df
        
        self.strikepoints = strikepoints

class SOLPSdata:
    def __init__(self):
        
        self.params = defaultdict(dict)
        self.omp = pd.DataFrame()
        self.imp = pd.DataFrame()
        self.code = "SOLPS"
        
        
    def read_last10s(self, casepath):
        """
        Read original last10s.pkl file
        """
        
        last10s = read_file(os.path.join(casepath, "last10s.pkl"))
        dfs = dict()
        
        for param in last10s.keys():
            if len(np.array(last10s[param]).shape) == 2: 
                if last10s[param].shape[1] == 2:
                    if any([param.endswith(x) for x in ["da", "dr", "di"]]):
                        dfs[param] = pd.DataFrame(last10s[param])
                        dfs[param].columns = ["pos", param]
                        dfs[param] = dfs[param].set_index("pos")
                        dfs[param].index = dfs[param].index.astype(float)
                        
                        if param.endswith("da"): #re.search(".*da", param):
                            self.params["omp"][param] = dfs[param]
                            
                        elif param.endswith("dr"):
                            self.params["outer_lower"][param] = dfs[param]
                            
                        elif param.endswith("di"):
                            self.params["imp"][param] = dfs[param]
        
        self.create_dataframes()
        
    def read_dataframes(self, path):
        """
        Read last10s.pkl re-saved as dataframes to avoid incompatibility
        """
        file = read_file(path)
        self.file = file
        
        # for region in self.params.keys():
        for param in self.file.keys():
            if any([x in param for x in ["ti3", "te3", "ne3", "dab23", "dmb23"]]):
                self.file[param].columns = ["pos", param]
                self.file[param] = self.file[param].set_index("pos")
                self.file[param].index = self.file[param].index.astype(float)
                
                if param.endswith("da"): #re.search(".*da", param):
                    self.params["omp"][param] = self.file[param]
                    
                elif param.endswith("dr"):
                    self.params["outer_lower"][param] = self.file[param]
                    
                elif param.endswith("di"):
                    self.params["imp"][param] = self.file[param]

                
        self.create_dataframes()
        
        
    def create_dataframes(self):
        self.regions = dict()
        self.regions["omp"] = pd.concat(self.params["omp"].values(), axis = 1)
        self.regions["imp"] = pd.concat(self.params["imp"].values(), axis = 1)
        self.regions["outer_lower"] = pd.concat(self.params["outer_lower"].values(), axis = 1)
        
        self.translate_names()
        
        
    def translate_names(self):
        # TODO Optimise this
        
        for region in self.regions.keys():
            df = self.regions[region]
            
            name_map = {
                "ti3":"Td+",
                "te3":"Te",
                "ne3":"Ne",
                "dnb23":"Nd",
                "tab23":"Td"
            }
            
            # If key in name_map, rename accordingly
            for key in name_map.keys():
                new_columns = []
                for col in df.columns:
                    if re.search(f"^{key}..", col):
                        new_columns.append(name_map[key])
                        # print(f"{name_map[key]} found: {col}")
                    else:
                        new_columns.append(col)
                df.columns = new_columns
        
    # def process_omp(self):
    #     self.regions["omp"] = self.regions["omp"].rename(columns = {
    #         "ti3da":"Td+",
    #         "te3da":"Te",
    #         "ne3da":"Ne",
    #         "dnb23di":"Nd",
    #         "dab23da":"ND", # As in SOLPS
    #         "dmb23da":"ND2" # As in SOLPS
    #     }
    #     )
        
    #     self.regions["omp"]["Nd"] = self.regions["omp"]["ND"] + self.regions["omp"]["ND2"]*2 # Combine atomic and molecular data to compare with Hermes-3
        
    
    # def process_imp(self):
    #     self.regions["imp"] = self.regions["imp"].rename(columns = {
    #         "ti3di":"Td+",
    #         "te3di":"Te",
    #         "ne3di":"Ne",
    #         "dnb23di":"Nd",
    #         "dab23di":"ND", # As in SOLPS
    #         "dmb23di":"ND2" # As in SOLPS
    #     }
    #     )
        
    #     # self.regions["imp"]["Nd"] = self.regions["imp"]["ND"] + self.regions["imp"]["ND2"]*2 # Combine atomic and molecular data to compare with Hermes-3
                
            
    # def process_target(self):
    #     self.regions["outer_lower"] = self.regions["outer_lower"].rename(columns = {
    #         "ti3dr":"Td+",
    #         "te3dr":"Te",
    #         "ne3dr":"Ne",
    #         "dnb23di":"Nd",
    #         "dab23dr":"ND", # As in SOLPS
    #         "dmb23dr":"ND2" # As in SOLPS
    #     }
    #     )
        
    #     self.regions["outer_lower"]["Nd"] = self.regions["outer_lower"]["ND"] + self.regions["outer_lower"]["ND2"]*2 # Combine atomic and molecular data to compare with Hermes-3

                
class Hermesdata:
    def __init__(self):
        self.code = "Hermes-3"
        pass
    
    def read_case(self, ds):

        self.ds = ds
        self.regions = dict()
        self.regions["omp"] = self.compile_results(ds.hermesm.select_region("outer_midplane_a"))
        self.regions["imp"] = self.compile_results(ds.hermesm.select_region("inner_midplane_a"))
        self.regions["outer_lower"] = self.compile_results(ds.hermesm.select_region("outer_lower_target"))
        
        self.regions["imp"].index *= -1    # Ensure core is on the LHS

        
    def compile_results(self, dataset):
        self.dataset = dataset

        params = ["Td+", "Td", "Te", "Ne", "Nd", "Sd+_iz"]
        x = []
        for param in params:
            data = self.dataset[param]
            df = pd.DataFrame(index = data["R"].data.flatten())
            df.index.name = "pos"
            df[param] = data.values
            x.append(df)
            
        df = pd.concat(x, axis = 1)

        # Normalise to separatrix
        sep_R = df.index[self.ds.metadata["ixseps1"]- self.ds.metadata["MXG"]]
        df.index -= sep_R
        
        return df
    
        
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
                 cmap = "Spectral_r"):
        

        self.param = param
        self.cases = cases
        num_cases = len(cases.keys())
        self.max, self.min = self.find_ranges()

        # Find ranges if not provided
        if vmin == None:
            vmin = self.min
        else:
            self.min = vmin
        if vmax == None:
            vmax = self.max
        else:
            self.max = vmax

        fig = plt.figure(dpi=dpi)
        fig.set_figheight(6)
        fig.set_figwidth(num_cases*4)

        # Plot grid
        gs0a = mpl.gridspec.GridSpec(
                                        ncols=num_cases+1, nrows=2,
                                        width_ratios = [1]*num_cases + [0.1], # cbar, empty space, control,
                                        height_ratios = [0.95, 0.05],
                                        wspace = 0.05
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
                print(casename)
                model["ds"][param].bout.polygon(ax = axes[i], 
                                                add_colorbar = False, logscale = logscale, 
                                                separatrix = True, cmap = cmap, 
                                                vmin = vmin, vmax = vmax, 
                                                antialias = False,

                                                linewidth = 0,
                                                )
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
                    
                    
                # plot_2d(fig, axes[i], where = model["path"], what = name_parser(param,"solps"), cmap = "Spectral_r", scale = ("log" if logscale is True else "linear"), vmin = vmin, vmax = vmax, plot_cbar = False)
                # axes[i].set_title(f"SOLPS\{casename}")
                
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
        axes[-1] = fig.add_subplot(gs0a[0,-1])   
        norm = xbout.plotting.utils._create_norm(logscale, None, vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(mappable = sm, cax = axes[-1], label = param)

        
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
    
def name_parser(x, code):
    
    solps = {
        "Ne" : "ne",
        "Te" : "te",
        "Td+" : "ti",   
        "Nd" : "pdenn",   # Combined atoms+molecules. Custom made by matteo. Atoms: pdena, Molecules: pdenm
        "Td" : "tdena"    # Compare only atom temperature, ignore molecules (more physical)
    }
    
    if code == "solps":
        return solps[x]
        

            
          
        
def file_write(data, filename):
# Writes an object to a pickle file.
    
    with open(filename, "wb") as file:
    # Open file in write binary mode, dump result to file
        pkl.dump(data, file)
        
        
        
def file_read(filename):
# Reads a pickle file and returns it.

    with open(filename, "rb") as filename:
    # Open file in read binary mode, dump file to result.
        data = pkl.load(filename)
        
    return data


def lineplot(
    cases,
    mode = "log",
    colors = ["black", "red", "black", "red", "navy", "limegreen", "firebrick",  "limegreen", "magenta","cyan", "navy"],
    params = ["Td+", "Te", "Td", "Ne", "Nd"],
    regions = ["imp", "omp", "outer_lower"],
    ylims = (None,None),
    dpi = 120
    ):
    
    marker = "o"
    ms = 0
    lw = 1.5
    set_ylims = dict()
    set_yscales = dict()


    if mode == "sol":
        
        set_ylims = {
        "imp"         : {"Td+": (0, 300), "Te": (0, 250), "Ne": (1e15, 3e19), "Nd": (0.15e10, 1e18)},
        "omp"         : {"Td+": (0, 250), "Te": (0, 250), "Ne": (1e17, 0.5e20), "Nd": (1e10, 1e16)},
        "outer_lower" : {"Td+": (0, 250), "Te": (0, 250), "Ne": (None, 4e18), "Nd": (0, None)},
        }
        
        set_yscales = {
        "imp" : {"Td+": "linear", "Te": "linear", "Ne": "linear", "Nd": "linear"},
        "omp" : {"Td+": "linear", "Te": "linear", "Ne": "linear", "Nd": "linear"},
        "outer_lower" : {"Td+": "linear", "Te": "linear", "Ne": "linear", "Nd": "log"},
        }
        
        xlims = (-0.01, None)
        
    if mode == "core":
        
        set_ylims = {
        "imp"         : {"Td+": (0, 2600), "Te": (0, 2500), "Ne": (1e15, 8e19), "Nd": (0.15e10, 1e16)},
        "omp"         : {"Td+": (0, 2600), "Te": (0, 2500), "Ne": (1e17, 1e20), "Nd": (1e10, 1e16)},
        "outer_lower" : {"Td+": (0, 250), "Te": (0, 250), "Ne": (None, 4e18), "Nd": (0, None)},
        }
        
        set_yscales = {
        "imp" : {"Td+": "linear", "Te": "linear", "Ne": "linear", "Nd": "linear"},
        "omp" : {"Td+": "linear", "Te": "linear", "Ne": "linear", "Nd": "linear"},
        "outer_lower" : {"Td+": "linear", "Te": "linear", "Ne": "linear", "Nd": "log"},
        }
        
        xlims = (None, 0.01)
        # xlims = (None, None)
        
        
    elif mode == "log":
        
        set_ylims = {
        "imp"         : {"Td+": (0, 2600), "Te": (0, 2600), "Ne": (1e15, 2e20), "Nd": (1e13, 0.15e18)},
        "omp"         : {"Td+": (0, 2600), "Te": (0, 2600), "Ne": (1e17, 2e20), "Nd": (1e14, 1e18)},
        "outer_lower" : {"Td+": (0, 250), "Te": (0, 250), "Ne": (None, 4e18), "Nd": (0, None)},
        }
            
        set_yscales = {
        "omp" : {"Td+": "log", "Te": "log", "Ne": "log", "Nd": "log"},
        "imp" : {"Td+": "log", "Te": "log", "Ne": "log", "Nd": "log"},
        "outer_lower" : {"Td+": "linear", "Te": "linear", "Td":"linear","Ne": "log", "Nd": "log"},
        }
        
        xlims = (None, None)



    for region in regions:

        fig, axes = plt.subplots(1,len(params), dpi = dpi, figsize = (4.2*len(params),5), sharex = True)
        fig.subplots_adjust(hspace = 0, wspace = 0.25, bottom = 0.25, left = 0.1, right = 0.9)

        
        for i, param in enumerate(params):
            for j, name in enumerate(cases.keys()):
                if region in cases[name].regions.keys():
                    data = cases[name].regions[region]
                    if param in data.columns:
                        if "SOLPS" in name:
                            ls = "--"
                        elif "SOLEDGE" in name:
                            ls = ":"
                        else:
                            ls = "-"
                        axes[i].plot(data.index, data[param], label = name, c = colors[j], marker = marker, ms = ms, lw = lw, ls = ls)
                    
            if param in set_yscales[region].keys():
                axes[i].set_yscale(set_yscales[region][param])
            else:
                if "T" in param or "N" in param and "outer_lower" not in region:
                    axes[i].set_yscale("log")
                
            

            ymin = []; ymax = []
            for name in cases.keys():
                if cases[name].code != "SOLEDGE2D":
                    ymin.append(cases[name].regions[region][param].min())
                    ymax.append(cases[name].regions[region][param].max())
            ymin = min(ymin)*0.8
            ymax = max(ymax)*1.2
            
            axes[i].set_ylim(ymin,ymax)
            # print(ymin)
            # print(ymax)
            
            # if param in set_ylims[region].keys():
            #     if set_ylims[region][param] != (None, None):
            #         axes[i].set_ylim(set_ylims[region][param])
                
             
            if ylims != (None, None):
                axes[i].set_ylim(ylims)
            if xlims != (None, None):
                axes[i].set_xlim(xlims)
            
            axes[i].grid(which="both", alpha = 0.2)
            axes[i].set_xlabel("Distance from separatrix [m]")
            # axes[i].legend()
            axes[i].set_title(f"{region}: {param}")
            
            if region == "omp":
                axes[i].set_xlim(-0.06, 0.05)
            elif region == "imp":
                axes[i].set_xlim(-0.11, 0.09)
            elif region == "outer_lower":
                axes[i].set_xlim(-0.05, 0.1)
            
        legend_items = []
        for j, name in enumerate(cases.keys()):
            if "SOLPS" in name:
                ls = "--"
            elif "SOLEDGE" in name:
                ls = ":"
            else:
                ls = "-"
            legend_items.append(mpl.lines.Line2D([0], [0], color=colors[j], lw=2, ls = ls))
            
        fig.legend(legend_items, cases.keys(), ncol = len(cases), loc = "upper center", bbox_to_anchor=(0.5,0.15))