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

from soledge.wrapper_class import *

from hermes3.utils import *

def save_last10s_subset(solps_path, destination_path):
    solps = file_read(solps_path)

    dfs = dict()
    for key in solps.keys():
        if any([x in key for x in ["ti3", "te3", "ne3", "dab23", "dmb23"]]):
            dfs[key] = pd.DataFrame(solps[key])

    file_write(dfs, destination_path)
    
    
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

class SOLEDGEdata:
    """
    Read and parse SOLEDGE parameters
    """

    def __init__(self, path = None):
        self.regions = dict()
        self.code = "SOLEDGE2D"
        if path != None:
            self.path = path
            self.case = SOLEDGEcase(path)
    
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
        df = df.rename(columns = {
                            'Dense' : "Ne", 
                            'Tempe':"Te", 
                            'Densi':"Nd+", 
                            'Tempi':"Td+",
                            'velocityi':"Vd+",
                            'Ppi':"Pd+",
                            'Ppe':"Pe",
                            'IRadi':"Rd+_ex",
                            "Nni":"Na",
                            "Nmi":"Nm",
                            "Tni":"Ta",
                            "Tmi":"Tm",
                            "Pni":"Pa",
                            "vyni":"Vyd",
                            "DIST":"x"})
        df = df.set_index("x")
        df.index.name = "pos"
        
        # Merge with existing data if it exists
        if name in self.regions:
            if all(df.index == self.regions[name].index):
                new_cols = df.columns.difference(self.regions[name].columns)
                self.regions[name] = pd.merge(self.regions[name], df[new_cols], left_index = True, right_index = True)
            else:
                raise Exception(f"Positions in the two {name} dataframes not matching")
        else:
            self.regions[name] = df
        
        
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
        
        # Add neutrals
        target = self.case.get_wall_data_on_target(self.case.get_wall_ntmpi(), "outer_lower")
        
        for col in target.columns:
            if col not in ["Ne", "Te"]:
                self.wallring[col] = target[col].values
                
        self.wallring = self.wallring.rename(columns = {
                            'Tn' : "Ta", 
                            'Nn' : "Na", 
                            })
        
        # Create copy of each df for each target with 0 being the strikepoint.
        for i, target in enumerate(["inner_lower", "outer_lower", "outer_upper", "inner_upper"]):
            strikepoints[target] = self.wallring.index[peaks[i]]
            df = self.wallring.copy()
            df.index -= strikepoints[target]
            
            self.regions[target] = df
            
        
        
        self.strikepoints = strikepoints
        
    def derive_variables(self):
        for region in self.regions.keys():
            if "Te" in self.regions[region].keys():
                self.regions[region]["Pe"] = self.regions[region]["Ne"] * self.regions[region]["Te"] * constants("q_e")
            if "Td+" in self.regions[region].keys():
                self.regions[region]["Pd+"] = self.regions[region]["Ne"] * self.regions[region]["Td+"] * constants("q_e")
            if "Ta" in self.regions[region].keys():
                self.regions[region]["Pa"] = self.regions[region]["Na"] * self.regions[region]["Ta"] * constants("q_e")
            if "Tm" in self.regions[region].keys():
                self.regions[region]["Pm"] = self.regions[region]["Nm"] * self.regions[region]["Tm"] * constants("q_e")
            

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
            if any([x in param for x in ["ti3", "te3", "ne3", "dab23", "dmb23", "rfluxa3", "refluxm3", "AMJUEL_H.4_2.1.5_3d"]]):
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
        self.regions["outer_upper"] = self.compile_results(ds.hermesm.select_region("outer_upper_target"))
        
        self.regions["imp"].index *= -1    # Ensure core is on the LHS
        
        # Parse column names to fit with the other codes
        for region in self.regions.keys():
            self.regions[region] = self.regions[region].rename(columns = {
                "Td" : "Ta",
                "Nd" : "Na",
                "Pd" : "Pa"
            })

        
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
    dpi = 120,
    lw = 1.5,
    set_xlim = True
    ):
    
    marker = "o"
    ms = 0
    set_ylims = dict()
    set_yscales = dict()

        
    set_yscales = {
    "omp" : {"Td+": "log", "Te": "log", "Ne": "log", "Nd": "log", "Pe":"log", "Pd+":"log"},
    "imp" : {"Td+": "log", "Te": "log", "Ne": "log", "Nd": "log", "Pe":"log", "Pd+":"log"},
    "outer_lower" : {"Td+": "linear", "Te": "linear", "Td":"linear","Ne": "linear", "Nd": "log"},
    "outer_upper" : {"Td+": "linear", "Te": "linear", "Td":"linear","Ne": "linear", "Nd": "log"},
    }
    
    region_extents = {
        "omp" : (-0.06, 0.05),
        "imp" : (-0.10, 0.06),
        "outer_lower" : (-0.02, 0.05),
        "outer_upper" : (-0.02, 0.05)
    }
    
    xlims = (None, None)



    for region in regions:

        scale = 1.1
        fig, axes = plt.subplots(1,len(params), dpi = dpi*scale, figsize = (4.2*len(params)/scale,5/scale), sharex = True)
        fig.subplots_adjust(hspace = 0, wspace = 0.3, bottom = 0.25, left = 0.1, right = 0.9)
        fig.suptitle(region, 
                     x = 0.06, y = 1.0, 
                     fontsize = "xx-large", color = "darkorchid", horizontalalignment = "left")
        
        linestyles = {"Hermes-3" : "-", "SOLEDGE2D" : ":", "SOLPS" : "--"}
        styles = {
            "Hermes-3" : {"ls" : "-"},
            "SOLEDGE2D" : {"ls" : "-", "lw" : 0, "marker" : "o", "ms" : 4, "markerfacecolor":"auto", "markeredgewidth":1},
            "SOLPS" : {"ls" : "-", "lw" : 0, "marker" : "x", "ms" : 3, "markeredgewidth":2}
        }


        ### For each parameter
        for i, param in enumerate(params):
            
            ymin = []; ymax = []
            
            ### For each case
            for j, name in enumerate(cases.keys()):
                case = cases[name]["data"]
                color = cases[name]["color"]
                code = case.code
                ls = linestyles[code]
                
                ## Find region
                if region in case.regions.keys():   # Is region available?
                    data = case.regions[region]
                    
                    ## Find parameter
                    if param in data.columns:   # If the parameter is available already, it's probaby custom made.
                        parsed_param = param
                    else:    # If it's not, then let's translate it. If not available, parsed_param is None
                        if code == "SOLPS":
                            parsed_param = parse_solps(param, region)
                        else:
                            parsed_param = None
                            
                    
                    ## Did we successfully parse the parameter?
                    if parsed_param != None and parsed_param in data.columns:    # Is parameter available?
                        
                        ## Crop SOLEDGE2D results to allow easier min/max finding
                        if code == "SOLEDGE2D":
                            data = data.query(f"(index > {region_extents[region][0]}) & (index < {region_extents[region][1]})")
                        
                        # print(f"Plotting {code}, {region}, {param}, {parsed_param}")
                        style_kwargs = styles[code]
                        axes[i].plot(data.index, data[parsed_param], label = name, c = color, **style_kwargs)
                        
                        # Collect min and max for later
                        ymin.append(case.regions[region][parsed_param].min())
                        ymax.append(case.regions[region][parsed_param].max())

                    ## Some issue with parameter
                    else:
                        print(f"{parsed_param} not available in {name}, {region}")
                    
            # Set yscales
            if param in set_yscales[region].keys():
                axes[i].set_yscale(set_yscales[region][param])
            else:
                if "T" in param or "N" in param and "outer_lower" not in region:
                    axes[i].set_yscale("log")
            xlims = (None,None)
            ylims = (None,None)
            # Set ylims
            if len(ymin) == 0:
                raise Exception(f"No data found for {param}")
            ymin = min(ymin)
            ymax = max(ymax)
            range = ymax - ymin
            
            axes[i].autoscale()
            
            
            
             
            if ylims != (None, None):
                axes[i].set_ylim(ylims)
                
            if set_xlim is True:
                
                if xlims != (None, None):
                    axes[i].set_xlim(xlims)
                    
                if region == "omp":
                    axes[i].set_xlim(-0.06, 0.05)
                elif region == "imp":
                    axes[i].set_xlim(-0.11, 0.09)
                elif region == "outer_lower":
                    axes[i].set_xlim(-0.03, 0.05 )
            
            axes[i].grid(which="both", color = "k", alpha = 0.05, lw = 0.5)
            
            ## Title and axis labels
            axes[i].set_title(f"{param}", fontsize = "xx-large")
            axes[i].set_xlabel("$X-X_{sep}$ [m]")
            
            # Set units in ylabel
            units = {
                "Ne":"$m^{-3}$", "Nd":"$m^{-3}$", "Nn":"$m^{-3}$",  "Na":"$m^{-3}$",  "Nm":"$m^{-3}$", 
                "Te":"eV", "Td+" : "eV", "Td" : "eV",
                "Pa":"Pa", "Pd+":"Pa", "Pe":"Pa"}
            
            if param in units:
                axes[i].set_ylabel(units[param]) 
                
            axes[i].xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=3, nbins=5))
            # axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=3, nbins=7))
            
            
        ### Make legend
        legend_items = []
        for j, name in enumerate(cases.keys()):
            if "SOLPS" in name:
                ls = linestyles["SOLPS"]
            elif "SOLEDGE" in name:
                ls = linestyles["SOLEDGE2D"]
            else:
                ls = linestyles["Hermes-3"]
                
            style_kwargs = styles[cases[name]["data"].code]
            legend_items.append(mpl.lines.Line2D([0], [0], color=cases[name]["color"], **style_kwargs))
            
        fig.legend(legend_items, cases.keys(), ncol = int(len(cases)), loc = "upper center", bbox_to_anchor=(0.5,0.10))
        # fig.legend(legend_items, cases.keys(), ncol = 1, loc = "upper right", bbox_to_anchor=(0.05,0.90))
        # fig.tight_layout()
        
            
          
        
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