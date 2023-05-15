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
    """
    Read and parse SOLEDGE parameters
    """

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

        self.derive_variables()
            
    def process_plot1d(self, df, name):
        """ 
        Process the csv file named "plot1d" which has radial profiles at the OMP.
        """
        self.regions[name] = df
        self.regions[name] = self.regions[name].rename(columns = {
                                            'Dense' : "Ne", 
                                            'Tempe':"Te", 
                                            'Densi':"Nd+", 
                                            'Tempi':"Td+",
                                            'velocityi':"Vd+",
                                            'Ppi':"Pd+",
                                            'Ppe':"Pe",
                                            'IRadi':"Rd+_ex",
                                            "Nni":"Nd",
                                            "Nmi":"Nd2",
                                            "Tni":"Td",
                                            "Tmi":"Td2",
                                            "Pni":"Pd",
                                            "vyni":"Vyd",
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
        
    def derive_variables(self):
        for region in self.regions.keys():
            self.regions[region]["Pe"] = self.regions[region]["Ne"] * self.regions[region]["Te"] * constants("q_e")
            self.regions[region]["Pd+"] = self.regions[region]["Ne"] * self.regions[region]["Td+"] * constants("q_e")
            

class SOLPSdata:
    def __init__(self):
        
        self.params = defaultdict(dict)
        self.omp = pd.DataFrame()
        self.imp = pd.DataFrame()
        self.code = "SOLPS"
        
        
    def read_last10s(self, casepath):
        """
        Read original last10s.pkl file
        da = OMP
        dr = outer lower target
        di = IMP
        """
        
        last10s = read_file(os.path.join(casepath, "last10s.pkl"))
        self.last10s = last10s
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
        self.derive_variables()
        
    def read_dataframes(self, path):
        """
        Read last10s.pkl re-saved as dataframes to avoid incompatibility
        """
        file = read_file(path)
        self.file = file
        
        # for region in self.params.keys():
        for param in self.file.keys():
            if any([x in param for x in ["ti3", "te3", "ne3", "dab23", "dmb23", "AMJUEL_H.4_2.1.5_3d"]]):
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
        

    def derive_variables(self):
        """
        Create custom variables
        These are generally defined in Hermes-3 like convention as the plotter checks
        for whether Hermes-3 like variable names exist first.
        """
        for region in self.regions.keys():
            
            def get_par(x):
                return self.regions[region][parse_solps(x, region)]
            
            self.regions[region]["Pe"] = get_par("Ne") * get_par("Te") * constants("q_e")
            self.regions[region]["Pd+"] = get_par("Ne") * get_par("Td+") * constants("q_e")
                
                
def parse_solps(param, loc):
    """
    Take Hermes-3 variable name and location
    Output SOLPS variable name
    """
    loc_key = {
        "omp" : "da",
        "imp" : "di",
        "outer_lower" : "dr",
    }
    
    param_key = {
        "Te" : "te3",
        "Td+" : "ti3",
        "Ne" : "ne3",
        "Nd" : "dab23",
        "Td" : "tab23",
        "Sd+_iz" : "AMJUEL_H.4_2.1.5_3" ,
    }
    
    if param not in param_key.keys():
        print(f"Parameter {param} not available in SOLPS")
        return None
    if loc not in loc_key.keys():
        print(f"Region {loc} not available in SOLPS")
        return None
    
    return f"{param_key[param]}{loc_key[loc]}"


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

        params = ["Td+", "Td", "Te", "Ne", "Pe", "Pd+", "Pd", "Nd", "Sd+_iz"]
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
    
        
    

        




def lineplot_compare(
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
        
        linestyles = {"Hermes-3" : "-", "SOLEDGE2D" : "dashdot", "SOLPS" : "--"}


        
        for i, param in enumerate(params):
            
            ymin = []; ymax = []
            
            for j, name in enumerate(cases.keys()):
                code = cases[name].code
                ls = linestyles[code]
                
                if region in cases[name].regions.keys():   # Is region available?
                    data = cases[name].regions[region]
                    
                    if param in data.columns:   # If the parameter is available already, it's probaby custom made.
                        parsed_param = param
                    else:    # If it's not, then let's translate it. If not available, parsed_param is None
                        if code == "SOLPS":
                            parsed_param = parse_solps(param, region)
                        else:
                            parsed_param = param
                    
                    
                    if parsed_param != None and parsed_param in data.columns:    # Is parameter available?
                        # print(f"Plotting {code}, {region}, {param}, {parsed_param}")
                        axes[i].plot(data.index, data[parsed_param], label = name, c = colors[j], marker = marker, ms = ms, lw = lw, ls = ls)
                        
                        # Collect min and max for later
                        if code != "SOLEDGE2D":
                            ymin.append(cases[name].regions[region][parsed_param].min())
                            ymax.append(cases[name].regions[region][parsed_param].max())
                    
            # Set yscales
            if param in set_yscales[region].keys():
                axes[i].set_yscale(set_yscales[region][param])
            else:
                if "T" in param or "N" in param and "outer_lower" not in region:
                    axes[i].set_yscale("log")
            xlims = (None,None)
            ylims = (None,None)
            # Set ylims
            ymin = min(ymin)*0.8
            ymax = max(ymax)*1.2
            
            if "Td+" in param:
                ymin *= 0.4
            if "Td" in param:
                ymin *= 0.9
            # if "Nd" in param:
            #     ymax *= 10

            axes[i].set_ylim(ymin,ymax) 
            

            # ymin = []; ymax = []
            # for name in cases.keys():
            #     if cases[name].code != "SOLEDGE2D":
            #         print(name)
            #         ymin.append(cases[name].regions[region][param].min())
            #         ymax.append(cases[name].regions[region][param].max())
            
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
                ls = linestyles["SOLPS"]
            elif "SOLEDGE" in name:
                ls = linestyles["SOLEDGE2D"]
            else:
                ls = linestyles["Hermes-3"]
            legend_items.append(mpl.lines.Line2D([0], [0], color=colors[j], lw=2, ls = ls))
            
        fig.legend(legend_items, cases.keys(), ncol = len(cases), loc = "upper center", bbox_to_anchor=(0.5,0.15))
        
        
            
          
        
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