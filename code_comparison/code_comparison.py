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
    print("Gridtools not found in code_comparison")

try:
    from soledge.wrapper_class import *
except:
    print("SOLEDGE could not be imported in code_comparison")

from hermes3.utils import *
from hermes3.selectors import get_1d_radial_data, get_1d_poloidal_data
from code_comparison.solps_pp import SOLPScase


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
        "Nd" : "pdena",
        "Na" : "pdena",
        "Nm" : "pdenm",   
        "Td" : "tdena",    
        "Ta" : "tdena", 
        "Tm" : "tdenm", 
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
    
    hermes = {
        "Rtot":"Rd+_"
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
            
        ### Get poloidal data
        self.regions["outer_fieldline"] = self.case.get_1d_poloidal_data(params = self.case.params, d_from_sep=0.0009)
        self.regions["outer_fieldline_parallel"] = self.case.get_1d_poloidal_data(params = self.case.params, d_from_sep=0.0009, parallel_length = True)
        
        
        self.regions["inner_fieldline_0.001_parallel"] = self.case.get_1d_poloidal_data(params = self.case.params, region =  "outer_lower", d_from_sep = 0.001, parallel_length = True)
        self.regions["outer_fieldline_0.001_parallel"] = self.case.get_1d_poloidal_data(params = self.case.params, region =  "outer_upper", d_from_sep = 0.001, parallel_length = True)
        self.regions["outer_fieldline_0.003_parallel"] = self.case.get_1d_poloidal_data(params = self.case.params, region =  "outer_upper", d_from_sep = 0.003, parallel_length = True)
        self.regions["outer_fieldline_0.015_parallel"] = self.case.get_1d_poloidal_data(params = self.case.params, region =  "outer_upper", d_from_sep = 0.015, parallel_length = True)
        self.regions["outer_fieldline_0.030_parallel"] = self.case.get_1d_poloidal_data(params = self.case.params, region =  "outer_upper", d_from_sep = 0.030, parallel_length = True)
        for region in self.regions.keys():
            if "fieldline" in region:
                
                
                df = self.regions[region]
                
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
                
                
                cs = np.sqrt((df["Te"] + df["Td+"]) * constants("q_e") / (constants("mass_p")*2))
                
                df["M"] = df["Mi"]  # Somehow M not available anymore but Vd+ is 
                df["Vd+"] = df["M"] * cs
                df["NVd+"] = df["Vd+"] * df["Ne"] * constants("mass_p") * 2

                df["dist"] = df["dist"][0] - df["dist"]
                
                df.index = df["dist"]
                df.index.name = "pos"
                # df = df.query("index > 0")
                self.regions[region] = df
                
                self.regions[region]["Nn"] = self.regions[region]["Na"] + self.regions[region]["Nm"]*2
                self.regions[region]["Pm"] = self.regions[region]["Tm"] * self.regions[region]["Nm"] * constants("q_e")
                self.regions[region]["Pn"] = self.regions[region]["Pa"] + self.regions[region]["Pm"]
                self.regions[region]["Tn"] = self.regions[region]["Pn"] / self.regions[region]["Nn"] / constants("q_e")
                
                
                
            if "inner" in region or "outer" in region:
                self.regions[region]["M"] = np.abs(self.regions[region]["M"])
                self.regions[region]["Vd+"] = np.abs(self.regions[region]["Vd+"])
                self.regions[region]["NVd+"] = np.abs(self.regions[region]["NVd+"])
                
        
                
           
    
    def read_csv(self, path, mode):
        
        df = pd.read_csv(path)
        
        if mode == "wall_ntmpi":
            self.wallring = df
            self.process_ring()
        
        if mode == "plot1d_omp":
            self.process_plot1d(df, "omp")
            
        elif mode == "plot1d_imp":
            self.process_plot1d(df, "imp")
            
        

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
        # if "imp" in name:
        #     df.index *= -1
        
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
                            ' Mach_i': "Md+",
                            ' Ioniz_H' : "Ioniz_H",
                            'l (m)' : "l"
                            })
        
        self.wallring["Ne"] *= 10**19
        self.wallring["Nd+"] *= 10**19
        self.wallring["M"] = np.abs(self.wallring["Md+"])
        self.wallring["NVd+"] = np.abs(self.wallring["Jsat_d+"] * 1e6) / constants("q_e") * constants("mass_p") * 2
        self.wallring["Vd+"] = self.wallring["NVd+"] / ((constants("mass_p")*2) * self.wallring["Ne"])
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
            
            # Inner targets are mirrored about the strikepoint
            if "inner" in target:
                df.index *= -1
            
        
        
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
            if "Vd+" in self.regions[region].keys():
                self.regions[region]["NVd+"] = self.regions[region]["Vd+"] * self.regions[region]["Ne"] * constants("mass_p") * 2
                    
            if "Na" in self.regions[region].keys():
                self.regions[region]["Nn"] = self.regions[region]["Na"] + self.regions[region]["Nm"]*2
                self.regions[region]["Pn"] = self.regions[region]["Pa"] + self.regions[region]["Pm"]
                self.regions[region]["Tn"] = self.regions[region]["Pn"] / self.regions[region]["Nn"] / constants("q_e")
            

class SOLPSdata:
    def __init__(self):
        
        self.params = defaultdict(dict)
        self.omp = pd.DataFrame()
        self.imp = pd.DataFrame()
        self.code = "SOLPS"
        
    def read_from_case(self, casepath):
        
        spc = SOLPScase(casepath)
        spc.derive_data()
        data = spc.bal
        
        regions = {}

        # index = np.cumsum(self.g["hx"][selector])
        list_params = ["Td+", "Te", "Ne", "Pe", "Pd+", "Na", "Nn", "Nm", "Ta", "Tn", "Tm", "Pa", "Pm", "Pn", "NVd+", "Vd+", "M"]

        ### ALL WITH NO GUARDS

        # OMP
        
        for name in ["omp", "imp", "outer_lower_target", "inner_lower_target"]:
            
            df = spc.get_1d_radial_data(list_params, region = name)
            df.index = df.pop("dist")
            translate = dict(omp="omp", imp="imp", outer_lower_target="outer_lower", inner_lower_target="inner_lower")
            
            regions[translate[name]] = df.copy()

            
        regions["inner_fieldline_0.001"] = spc.get_1d_poloidal_data(list_params, region =  "inner_lower", sepdist = 0.001)
        regions["outer_fieldline"] = spc.get_1d_poloidal_data(list_params, region =  "outer_lower", sepdist = 0.001)
        regions["outer_fieldline_0.001"] = spc.get_1d_poloidal_data(list_params, region =  "outer_lower", sepdist = 0.001)
        regions["outer_fieldline_0.003"] = spc.get_1d_poloidal_data(list_params, region =  "outer_lower", sepdist = 0.003)
        regions["outer_fieldline_0.015"] = spc.get_1d_poloidal_data(list_params, region =  "outer_lower", sepdist = 0.015)
        regions["outer_fieldline_0.030"] = spc.get_1d_poloidal_data(list_params, region =  "outer_lower", sepdist = 0.030)
        
        regions["inner_fieldline"] = spc.get_1d_poloidal_data(list_params, region =  "inner_lower", sepdist = 0.0014)
        
        for region in list(regions.keys()):
            if "fieldline" in region:
                regions[region+"_parallel"] = regions[region].copy()
            
        for region in regions.keys():
            if "fieldline" in region:
                if "parallel" in region:
                    regions[region].index = regions[region]["Spar"]
                else:
                    regions[region].index = regions[region]["Spol"] 
                    
                for param in ["Vd+", "NVd+"]:
                    regions[region][param] = regions[region][param] * -1
                    
            ## ABSOLUTE MOMENTUM AT RADIAL SURFACES
            if "outer_lower" in region or "inner_lower" in region:
                for param in ["Vd+", "NVd+", "M"]:
                    regions[region][param] = np.abs(regions[region][param])
                    
            ## MULTIPLY BY -1 FOR Inner LEG
            if "inner_fieldline" in region:
                for param in ["Vd+", "NVd+"]:
                    regions[region][param] = regions[region][param] * -1
        
        self.regions = regions
        
        
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
        "Na" : "dab23",
        "Nm" : "dmb23",
        "Ta" : "tab23",
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
        self.params =  [
            "Td+", "Td", "Te", "Ne", "Pe", "Pd+", "Pd", "Nd", "NVd+", "Vd+", "M",
            "Sd+_iz", "Rd+_ex", "Rd+_rec"]
        pass
    
    def read_case(self, ds):
        # NOTE: please provide dataset with guard cells. The selections below are incosnsitent, targets already 
        # missing guards

        self.ds = ds
        self.regions = dict()
        
        ## OLD
        # self.regions["omp"] = self.get_radial_data(ds.hermesm.select_region("outer_midplane_a").isel(x = slice(2,-2)))
        # self.regions["imp"] = self.get_radial_data(ds.hermesm.select_region("inner_midplane_a").isel(x = slice(2,-2)))
        # self.regions["outer_lower"] = self.get_radial_data(ds.hermesm.select_region("outer_lower_target"))
        # self.regions["outer_upper"] = self.get_radial_data(ds.hermesm.select_region("outer_upper_target"))
        # self.regions["inner_lower"] = self.get_radial_data(ds.hermesm.select_region("inner_lower_target"))
        # self.regions["outer_fieldline"] = self.get_poloidal_data()
        
        self.regions["omp"] = get_1d_radial_data(ds, self.params, region = "outer_midplane")
        self.regions["imp"] = get_1d_radial_data(ds, self.params, region = "inner_midplane")
        self.regions["outer_lower"] = get_1d_radial_data(ds, self.params, region = "outer_lower_target")
        self.regions["inner_lower"] = get_1d_radial_data(ds, self.params, region = "inner_lower_target")
        
        self.regions["inner_fieldline_0.001"] = get_1d_poloidal_data(ds, self.params, region =  "inner_lower", sepdist = 0.001)
        self.regions["outer_fieldline_0.001"] = get_1d_poloidal_data(ds, self.params, region =  "outer_lower", sepdist = 0.001)
        self.regions["outer_fieldline_0.003"] = get_1d_poloidal_data(ds, self.params, region =  "outer_lower", sepdist = 0.003)
        self.regions["outer_fieldline_0.015"] = get_1d_poloidal_data(ds, self.params, region =  "outer_lower", sepdist = 0.015)
        self.regions["outer_fieldline_0.030"] = get_1d_poloidal_data(ds, self.params, region =  "outer_lower", sepdist = 0.030)
        
        regions = list(self.regions.keys())
        for region in regions:
            if "fieldline" in region:
                self.regions[region+"_parallel"] = self.regions[region].copy()
                
            self.regions[region]["M"] = np.abs(self.regions[region]["M"])
            
        for region in list(self.regions.keys()):
            if "fieldline" in region:
                if "parallel" in region:
                    self.regions[region].index = self.regions[region]["Spar"]
                else:
                    self.regions[region].index = self.regions[region]["Spol"] 
                
                # Flip sign of flow related diagnostics for inner    
                if "inner" in region:
                    for param in ["Vd+", "NVd+", "M"]:
                        self.regions[region][param] = self.regions[region][param] * -1

        
        # Make distance the index
        for region in self.regions.keys():
            if "fieldline" not in region:
                self.regions[region].index = self.regions[region].pop("Srad")
            
        
        # Parse column names to fit with the other codes
        for region in self.regions.keys():
            self.regions[region] = self.regions[region].rename(columns = {
                "Td" : "Ta",
                "Nd" : "Na",
                "Pd" : "Pa"
            })
            
            self.regions[region]["Nn"] = self.regions[region]["Na"]
            self.regions[region]["Pn"] = self.regions[region]["Pa"]
            self.regions[region]["Tn"] = self.regions[region]["Ta"]

        
    def get_radial_data(self, dataset):
        """
        Deprecated
        """
        self.dataset = dataset

        x = []
        for param in self.params:
            
            ds = self.dataset   
            dr = np.cumsum(ds["dr"].values.flatten())
            m = self.ds.metadata
            sep_idx = m["ixseps1"] - m["MXG"]
            dist = dr - dr[sep_idx]
            
            df = pd.DataFrame(index = dist)
            df.index.name = "pos"
            if param in ds.data_vars:
                df[param] = ds[param]
            else:
                df[param] = np.nan
            x.append(df)
            
        df = pd.concat(x, axis = 1)

        # Normalise to separatrix
        sep_R = df.index[self.ds.metadata["ixseps1"]- self.ds.metadata["MXG"]]
        df.index -= sep_R
        
        return df
    
    def get_poloidal_data(self):
        """
        Deprecated
        """
        ds = self.ds
        m = ds.metadata
    
        # Find the right poloidal flux tube by looking at the midplane
        sep_dist = 0   # Hardcoded to separatrix
        omp = ds.hermesm.select_region("outer_midplane_a")
        
        if ds.dims["x"] == ds.metadata["nxg"]:
            adder = 0
        else:
            adder = 2
        
        id_sep = m["ixseps1"] - adder

        r = np.cumsum(omp["dr"].values)
        r = r - r[id_sep]

        id_fl = np.argmin(np.abs(r - sep_dist))
        id_omp = int((m["j2_2g"] - m["j1_2g"]) / 2) + m["j1_2g"]
        fl = ds.isel(theta = slice(id_omp, -m["MYG"]), x = id_fl)

        dist = np.cumsum(fl["dl"].values)
        dist -= dist[0]

        df = pd.DataFrame(index = dist)


        for param in self.params:
            if param in fl.data_vars:
                df[param] = fl[param].values
            else:
                df[param] = np.nan
            
        return df
    

        




def lineplot_compare(
    cases,
    mode = "log",
    colors = ["black", "red", "black", "red", "navy", "limegreen", "firebrick",  "limegreen", "magenta","cyan", "navy"],
    params = ["Td+", "Te", "Ta", "Ne", "Nd"],
    regions = ["imp", "omp", "outer_lower", "outer_fieldline_parallel"],
    ylims = (None,None),
    dpi = 120,
    lw = 1.5,
    set_xlim = True,
    legend_nrows = 2,
    combine_molecules = False,
    solps_noD2 = False,
    titles = "long",
    title_append = ""
    
    ):
    
    marker = "o"
    ms = 0
    set_ylims = dict()
    set_yscales = dict()

        
    set_yscales = {
    "omp" : {
        "Td+": "log", "Te": "log", "Ta" : "log", "Tm" : "log", "Tn": "log",
        "Ne": "log", "Nd": "log", "Na": "log", "Nm": "log", "Nn": "log",
        "Pe":"log", "Pd+":"log", "NVd+":"log", "Vd+": "log", "M": "linear",
        },
    "imp" : {
        "Td+": "log", "Te": "log", "Ta" : "log", "Tm" : "log",
        "Ne": "log", "Nd": "log", "Na" : "log", "Nm": "log", 
        "Pe":"log", "Pd+":"log", "NVd+":"log", "Vd+": "log", "M": "linear"},
    "outer_lower" : {"Td+": "linear", "Te": "linear", "Td":"linear","Ta":"linear", "Ne": "linear", "Nd": "log"},
    "outer_upper" : {"Td+": "linear", "Te": "linear", "Td":"linear","Ta":"linear", "Ne": "linear", "Nd": "log"},
    "outer_fieldline" : {"Td+": "linear", "Te": "linear", "Td":"linear","Ta":"linear", "Ne": "log", "Nd": "log"},
    "outer_fieldline_parallel" : {
        "Td+": "linear", "Te": "linear", "Td":"linear","Ta":"linear", "Tn":"linear","Tm":"linear",
        "Ne": "log", "Nd": "log", "Na":"log", "Nm":"log", "Nn":"log",
        "Pe": "linear", "Pd+": "linear", "Pa": "log", "Pn":"log", "Pm":"log",
        "M": "linear", "NVd+": "linear", "Vd+": "linear"},
    }
    set_yscales["inner_lower"] = set_yscales["outer_lower"]
    set_yscales["inner_upper"] = set_yscales["outer_upper"]
    
    region_extents = {
        "omp" : (-0.06, 0.06),
        "imp" : (-0.10, 0.06),
        "outer_lower" : (-0.018, 0.058),
        "outer_upper" : (-0.02, 0.05),
        "inner_lower" : (-0.04, 0.09),
        "outer_fieldline" : (0, 2),
        "outer_fieldline_parallel" : (0, 7),
        "inner_fieldline_parallel" : (0, 7)
    }
    
    xlims = (None, None)



    for region in regions:

        scale = 1.3
        fig, axes = plt.subplots(1,len(params), dpi = dpi*scale, figsize = (4.7*len(params)/scale,5/scale), sharex = True)
        fig.subplots_adjust(hspace = 0, wspace = 0.35, bottom = 0.25, left = 0.1, right = 0.9)
        fig.suptitle(region + title_append, x = 0.1, y = 1.0, fontsize = "xx-large", ha = "left")
        
        linestyles = {"Hermes-3" : "-", "SOLEDGE2D" : ":", "SOLPS" : "--"}
        styles = {
            "Hermes-3" : {"ls" : "-", "lw" : lw},
            "SOLEDGE2D" : {"ls" : "-", "lw" : 0, "marker" : "x", "ms" : 5, "markerfacecolor":"auto", "markeredgewidth":1, "zorder":100},
            "SOLPS" : {"ls" : "-", "lw" : 0, "marker" : "o", "ms" : 3, "markeredgewidth":0}
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
                    
                    if "fieldline" not in region:
                        xmult = 100
                    else:
                        xmult = 1
                    
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
                        
                        # Crop SOLEDGE2D results to allow easier min/max finding
                        if code == "SOLEDGE2D":
                            if "fieldline" not in region:
                                data = data.query(f"(index > {region_extents[region][0]}) & (index < {region_extents[region][1]})")
                            
                        
                        ### Molecule combination
                        atom_override = {}
                        
                        if combine_molecules is True:
                            
                            atom_alpha = 1
                            
                            if code == "SOLEDGE2D":
                                molstyle = {"lw" : 0, "marker" : "x","color":color,  "ms" : 5, "markeredgewidth":1, "zorder":101}
                                atomstyle = {"lw" : 0, "marker" : "*","color":color,  "ms" : 5, "markeredgewidth":1, "zorder":101}
                                
                                ## Partial pressures - Pt = Pa + Pm, no factors of 2.
                                if parsed_param == "Na":
                                    axes[i].plot(data.index*xmult, data["Na"] + data["Nm"], label = name, **molstyle)
                                # if parsed_param == "Pa":
                                #     axes[i].plot(data.index*xmult, data["Pa"]*0 + data["Pm"], label = name, **molstyle)
                                if parsed_param == "Ta":
                                    weighted_temp = ((data["Ta"] * data["Na"]) + (data["Tm"] * data["Nm"])) / (data["Na"] + data["Nm"]*2)
                                    axes[i].plot(data.index*xmult, weighted_temp, label = name, **molstyle) 
                                
                                # Make atoms grey
                                if any([x in parsed_param for x in ["Na", "Ta"]]):
                                    atom_override = dict(alpha = atom_alpha, marker = "o", markerfacecolor = "None")
                                    
                            if code == "SOLPS" and solps_noD2 is False:
                                molstyle = {"lw" : 0, "marker" : "o", "color":color, "ms" : 3, "markeredgewidth":1, "zorder":101}
                                
                                if param == "Na":
                                    # parsed_atom = parse_solps("Na", region)
                                    # parsed_mol = parse_solps("Nm", region)

                                    axes[i].plot(data.index*xmult, data["Na"] + data["Nm"], label = name, **molstyle)
                                    
                                if param == "Ta":

                                    weighted_temp = ((data["Ta"] * data["Na"]) + (data["Tm"] * data["Nm"])) / (data["Na"] + data["Nm"]*2)
                                    axes[i].plot(data.index*xmult, weighted_temp, label = name, **molstyle)
                                
                                # Make atoms grey
                                if any([x in parsed_param for x in ["Na", "Ta"]]):
                                    atom_override = dict(alpha = atom_alpha, marker = "o")
                        
                        # print(f"Plotting {code}, {region}, {param}, {parsed_param}")
                        input_dict = cases[name]
                        custom_kwargs = {}
                        for val in input_dict:
                            if val != "data": custom_kwargs[val] = input_dict[val]
                        style_kwargs = {**styles[code], **custom_kwargs, **atom_override}
                        style_kwargs = {**style_kwargs, **atom_override}
                        
                        axes[i].plot(data.index*xmult, data[parsed_param], label = name,  **style_kwargs)
                        
                        # Collect min and max for later
                        ymin.append(case.regions[region][parsed_param].min())
                        ymax.append(case.regions[region][parsed_param].max())

                    ## Some issue with parameter
                    else:
                        print(f"{parsed_param} not available in {name}, {region}")
                else:
                    print(f"{region} not available in {name}")
                    
            # Set yscales
            
                
            
            if "fieldline" in region:
                if param in set_yscales["outer_fieldline_parallel"].keys():
                    axes[i].set_yscale(set_yscales["outer_fieldline_parallel"][param])
                else:
                    axes[i].set_yscale("linear")
                    # raise Exception(f"Parameter {param} not available in yscales of region {region}")
                  
                
            elif region in set_yscales.keys():
                if param in set_yscales[region].keys():
                    axes[i].set_yscale(set_yscales[region][param])
                else:
                    axes[i].set_yscale("linear")
                    # raise Exception(f"Parameter {param} not available in yscales of region {region}")

                    
            if mode == "linear":
                axes[i].set_yscale("linear")
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
                    axes[i].set_xlim(-0.06*xmult, 0.05*xmult)
                elif region == "imp":
                    axes[i].set_xlim(-0.11*xmult, 0.09*xmult)
                elif region == "outer_lower":
                    axes[i].set_xlim(-0.03*xmult, 0.065*xmult )
                elif region == "inner_lower":
                    axes[i].set_xlim(-0.06*xmult, 0.10*xmult )
            
            axes[i].grid(which="both", color = "k", alpha = 0.1, lw = 0.5)
            
            if titles == "short":
                title_translate = {
                    "Td+" : "$T_{i}$", "Te" : "$T_{e}$", "Td+" : "$T_{i}$", "Ta" : "$T_{a}$",
                    "Ne" : "$N_{e}$", "Na" : "$N_{a}$",
                    "Pd+" : "$P_{i}$", "Pe" : "$P_{e}$", "Pd+" : "$P_{i}$",
                }
                title_fontsize = "xx-large"
                
            elif titles == "long":
                title_translate = {
                    "Td+" : "Ion temperature", "Te" : "Electron temperature", "Ta" : "Atom temperature", "Tm" : "Molecule temperature", "Tn" : "Neutral temperature",
                    "Ne" : "Electron density", "Na" : "Atom density", "Nm" : "Molecule density", "Nn" : "Neutral density",
                    "Pd+" : "Ion pressure", "Pe" : "Electron pressure", "Pa" : "Atom pressure", "Pm" : "Molecule pressure", "Pn" : "Neutral pressure"
                }
                title_fontsize = "large"
            
            
            title = title_translate[param] if param in title_translate.keys() else param
            
            ## Title and axis labels
            axes[i].set_title(loc = "right", y = 1.0, 
                              label = f"{title}", fontsize = title_fontsize, fontweight = "normal",
                              alpha = 1.0, color = "black")
            
            if "fieldline" in region:
                if "parallel" in region:
                    xlabel = r"$S_{\parallel} [m]$"
                else:
                    xlabel = r"$S_{\theta} [m]$"
            else:
                xlabel = "$X-X_{sep}$ [cm]"
            
            axes[i].set_xlabel(xlabel)
            
            # Set units in ylabel
            units = {
                "Ne":"$m^{-3}$", "Nd":"$m^{-3}$", "Nn":"$m^{-3}$",  "Na":"$m^{-3}$",  "Nm":"$m^{-3}$", 
                "Te":"eV", "Td+" : "eV", "Td" : "eV", "Ta":"eV", "Tm":"eV", "Ti":"eV",
                "Pa":"Pa", "Pd+":"Pa", "Pe":"Pa"}
            
            if param in units:
                axes[i].set_ylabel(units[param]) 
                
            axes[i].xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=3, nbins=5))
            # axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=3, nbins=7))
            
            ### Set different background for neutrals
            if param in  ["Nd", "Na", "Nn", "Nm", "Td", "Ta", "Tm", "Pd", "Pa", "Pm"]:
                axes[i].patch.set_facecolor("dodgerblue")
                axes[i].patch.set_alpha(0.02)
            elif param in ["Ne", "Nd+",  "Te", "Td+", "Pe", "Pd+"]:
                axes[i].patch.set_facecolor("darkorange")
                axes[i].patch.set_alpha(0.02)
            
            
        ### Make legend
        legend_items = []
        legend_labels = []
        for j, name in enumerate(cases.keys()):
            if "SOLPS" in name:
                ls = linestyles["SOLPS"]
            elif "SOLEDGE" in name:
                ls = linestyles["SOLEDGE2D"]
            else:
                ls = linestyles["Hermes-3"]
                
            style_kwargs = styles[cases[name]["data"].code]
            legend_items.append(mpl.lines.Line2D([0], [0], color=cases[name]["color"], **style_kwargs))
            legend_labels.append(name)
            if "SOLEDGE" in name and combine_molecules is True:
                legend_items.append(mpl.lines.Line2D([0], [0], color=cases[name]["color"], **{**style_kwargs, **dict(marker = "o", markerfacecolor = "None")}))
                legend_labels.append(f"{name} (atoms)")

        fig.legend(legend_items, legend_labels, ncol = int(len(cases)/legend_nrows), 
                   loc = "upper center", bbox_to_anchor=(0.5,0.10),
                   fontsize = "large", frameon = False, labelcolor ="darkslategrey")
        # fig.legend(legend_items, cases.keys(), ncol = 1, loc = "upper right", bbox_to_anchor=(0.05,0.90))
        # fig.tight_layout()
        
def plot_by_region(
    cases,
    regions,
    params,
    data_dicts,
    densities = ["1e19", "2e19", "3e19"],
    **kwargs):
    """
    Produces comparison plots grouped by region across all three densities. 
    The inputs MUST be in terms of 1e19 with 1e19 in the case name.
    
    e.g.
    
                   
    plot_by_region(
        cases = { 

            "SOLPS, phi=0" : dict(name="tightwall_noD2_1e19_zerophi", color="deeppink"),
            "SOLEDGE2D" : dict(name="tightwall_1e19", color="black"),
            r"Hermes-3: max_mfp=1, cond_alpha=0.25" : dict(name='1e19', color = "teal"),

        },
        regions = ["omp", "outer_fieldline_0.001_parallel", "outer_lower"],
        params = ["Ne", "Te", "Td+", "NVd+", "Vd+", "M"],
        data_dicts = {"SOLPS":sp, "SOLEDGE2D":sl, "Hermes-3":hr},
        dpi = 100,
        ylims = (5e17, 2e20),
        lw = 2,
        legend_nrows =1,
        combine_molecules = False)
    """
    
    ## Asssemble cases for all three densities
    cases_data = {}
    
    for dens in densities:
        cases_data[dens] = {}
        
        for casename, entry in cases.items():
            
            
            if "SOLPS" in casename:
                cases_data[dens][casename] = dict(data = data_dicts["SOLPS"][entry["name"].replace("1e19", dens)], color = entry["color"])
            elif "SOLEDGE" in casename:
                cases_data[dens][casename] = dict(data = data_dicts["SOLEDGE2D"][entry["name"].replace("1e19", dens)], color = entry["color"])
            elif "Hermes" in casename:
                cases_data[dens][casename] = dict(data = data_dicts["Hermes-3"][entry["name"].replace("1e19", dens)], color = entry["color"])
            else:
                raise ValueError(f"Unknown case type for {casename}")
                
                
    for region in regions:
        for dens in densities:
            lineplot_compare(
                cases = cases_data[dens],
                regions = [region],
                params = params,
                title_append = f" - {dens}",
                **kwargs
            )  
          
        
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