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

import gridtools.solps_python_scripts.setup
from gridtools.solps_python_scripts.plot_solps       import plot_1d, plot_2d

sys.path.append(r'C:\Users\mikek\OneDrive\Project\python-packages')
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
        
        self.process_omp()
        self.process_imp()
        self.process_target()
        
        
    def process_omp(self):
        self.regions["omp"] = self.regions["omp"].rename(columns = {
            "ti3da":"Td+",
            "te3da":"Te",
            "ne3da":"Ne",
            "dab23da":"ND", # As in SOLPS
            "dmb23da":"ND2" # As in SOLPS
        }
        )
        
        self.regions["omp"]["Nd"] = self.regions["omp"]["ND"] + self.regions["omp"]["ND2"]*2 # Combine atomic and molecular data to compare with Hermes-3
        
    
    def process_imp(self):
        self.regions["imp"] = self.regions["imp"].rename(columns = {
            "ti3di":"Td+",
            "te3di":"Te",
            "ne3di":"Ne",
            "dab23di":"ND", # As in SOLPS
            "dmb23di":"ND2" # As in SOLPS
        }
        )
        
        self.regions["imp"]["Nd"] = self.regions["imp"]["ND"] + self.regions["imp"]["ND2"]*2 # Combine atomic and molecular data to compare with Hermes-3
                
            
    def process_target(self):
        self.regions["outer_lower"] = self.regions["outer_lower"].rename(columns = {
            "ti3dr":"Td+",
            "te3dr":"Te",
            "ne3dr":"Ne",
            "dab23dr":"ND", # As in SOLPS
            "dmb23dr":"ND2" # As in SOLPS
        }
        )
        
        self.regions["outer_lower"]["Nd"] = self.regions["outer_lower"]["ND"] + self.regions["outer_lower"]["ND2"]*2 # Combine atomic and molecular data to compare with Hermes-3

                
class Hermesdata:
    def __init__(self):
        pass
    
    def read_case(self, case, tind = -1):
        self.case = case
        
        self.regions = dict()
        self.regions["omp"] = self.compile_results(self.case.select_region("outer_midplane_a").isel(t=tind))
        self.regions["imp"] = self.compile_results(self.case.select_region("inner_midplane_a").isel(t=tind))
        self.regions["outer_lower"] = self.compile_results(self.case.select_region("outer_lower_target").isel(t=tind))
        
        self.regions["imp"].index *= -1    # Ensure core is on the LHS

        
    def compile_results(self, dataset):
        self.dataset = dataset

        params = ["Td+", "Te", "Ne", "Nd", "Sd+_iz"]
        x = []
        for param in params:
            data = self.dataset[param]
            df = pd.DataFrame(index = data["R"].data.flatten())
            df.index.name = "pos"
            df[param] = data.values
            x.append(df)
            
        df = pd.concat(x, axis = 1)

        # Normalise to separatrix
        sep_R = df.index[self.case.ixseps1 + self.case.MXG]
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
                plot_2d(fig, axes[i], where = model["path"], what = name_parser(param,"solps"), cmap = "Spectral_r", scale = ("log" if logscale is True else "linear"), vmin = vmin, vmax = vmax, plot_cbar = False)
                axes[i].set_title(f"SOLPS\{casename}")
                
            axes[i].set_ylim(-0.88,0.1)
            axes[i].set_xlim(0.15, 0.78)
            
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
        "Nd" : "pdenn"   # Custom made by matteo. Atoms: pdena, Molecules: pdenm         
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