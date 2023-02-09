import numpy as np
import os, sys
import pandas as pd
import scipy
import re

sys.path.append(r'C:\Users\mikek\OneDrive\Project\python-packages')
from hermes3.utils import *



class SOLEDGEdata:

    def __init__(self):
        pass
    
    def read_csv(self, path, mode):
        
        df = pd.read_csv(path)
        
        if mode == "plot1d":
            self.omp = df
            self.process_omp()
            
        if mode == "wall_ntmpi":
            self.wallring = df
            self.process_ring()

            
    def process_omp(self):
        """ 
        Process the csv file named "plot1d" which has radial profiles at the OMP.
        """
        
        self.omp = self.omp.rename(columns = {'Dense' : "Ne", 
                                            'Tempe':"Te", 
                                            'Densi':"Nd+", 
                                            'Tempi':"Td+",
                                            'velocityi':"Vd+",
                                            'Ppi':"Pd+",
                                            'IRadi':"Rd+_ex",
                                            "DIST":"x"})
        self.omp = self.omp.set_index("x")
        self.omp.index.name = "pos"
        
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
            
            setattr(self, target, df)
        
        self.strikepoints = strikepoints

class SOLPSdata:
    def __init__(self):
        pass
    
    def read_dataframes(self, path):
        file = read_file(path)
        self.file = file
        self.params = file.keys()
        self.omp = pd.DataFrame()
        self.omp_params = [] #["ti3da", "te3da"]
        self.outer_lower_params = []

        
        for param in self.params:
            # print(self.file[param].columns())
            self.file[param].columns = ["pos", param]
            self.file[param] = self.file[param].set_index("pos")
            self.file[param].index = self.file[param].index.astype(float)
            
            if param.endswith("da"): #re.search(".*da", param):
                self.omp_params.append(self.file[param])
                
            elif param.endswith("dr"):
                self.outer_lower_params.append(self.file[param])
                
        self.omp = pd.concat(self.omp_params, axis = 1)
        self.outer_lower = pd.concat(self.outer_lower_params, axis = 1)
        self.process_omp()
        self.process_target()
        
        
    def process_omp(self):
        self.omp = self.omp.rename(columns = {
            "ti3da":"Td+",
            "te3da":"Te",
            "ne3da":"Ne",
            "dab23da":"ND", # As in SOLPS
            "dmb23da":"ND2" # As in SOLPS
        }
        )
        
        self.omp["Nd"] = self.omp["ND"] + self.omp["ND2"]*2 # Combine atomic and molecular data to compare with Hermes-3
                
            
    def process_target(self):
        self.outer_lower = self.outer_lower.rename(columns = {
            "ti3dr":"Td+",
            "te3dr":"Te",
            "ne3dr":"Ne",
            "dab23dr":"ND", # As in SOLPS
            "dmb23dr":"ND2" # As in SOLPS
        }
        )
        
        self.outer_lower["Nd"] = self.outer_lower["ND"] + self.outer_lower["ND2"]*2 # Combine atomic and molecular data to compare with Hermes-3
                
class Hermesdata:
    def __init__(self):
        pass
    
    def read_case(self, case):
        self.case = case
        
        self.omp = self.compile_results(self.case.select_region("outer_midplane_a").isel(t=-1))
        self.outer_lower = self.compile_results(self.case.select_region("outer_lower_target").isel(t=-1))

        
    def compile_results(self, dataset):
        self.dataset = dataset

        params = ["Td+", "Te", "Ne", "Nd"]
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
    
    def process_omp(self):
        self.omp_object = self.case.select_region("outer_midplane_a").isel(t=-1)

        self.omp = pd.DataFrame()

        params = ["Td+", "Te", "Ne", "Nd"]
        x = []
        for param in params:
            data = self.omp_object[param]
            df = pd.DataFrame(index = data["R"])
            df.index.name = "pos"
            df[param] = data.values
            x.append(df)
            
        self.omp = pd.concat(x, axis = 1)
        
        # Normalise to separatrix
        sep_R = self.omp.index[self.case.ixseps1 + self.case.MXG]
        self.omp.index -= sep_R
        
          
        
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