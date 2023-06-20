#!/usr/bin/env python3

from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, pathlib
import traceback
import platform
from scipy import stats
from boututils.datafile import DataFile
from boutdata.collect import collect
from boutdata.data import BoutData
import xbout
from .sk_plotting_functions_new import *


class SKDeck:
    def __init__(self, path_deck, key = "", keys = [], load_all = True):
        self.casepaths = []
        self.casenames = []
        self.timesteps = []

        for casepath in pathlib.Path(path_deck).rglob("DENS_INPUT.txt"):
            dir_run = pathlib.Path(casepath).parents[1]
            dir_case = pathlib.Path(casepath).parents[2]
            dir_output = os.path.join(dir_run, "OUTPUT")
            dir_density = os.path.join(dir_output, "DENSITY")
            file_density = os.listdir(dir_density)[-1]
            timestep = file_density.replace(".txt","").split("_")[-1]

            self.casepaths.append(dir_run)
            self.casenames.append(os.path.split(dir_case)[-1])
            self.timesteps.append(timestep)

        if load_all:
            self.load_cases()
            self.get_stats()
            
            
            for i, casename in enumerate(self.casenames):
                self.cases[casename].sk = self.read_sk(self.casepaths[i], self.timesteps[i])

    def load_cases(self):
        self.cases = dict()

        for i, casename in enumerate(self.casenames):
            self.cases[casename] = SKRun(self.casepaths[i])
            case = self.cases[casename]
            case.load_all()
            case.load_grids()
            case.calc_pressure()
            
            case.load_avg_density()
            case.target_flux, case.target_temp = case.get_target_conditions()
                    

    def get_stats(self):
        self.stats = pd.DataFrame()
        self.stats.index.rename("case", inplace = True)

        for casename in self.casenames:
            case = self.cases[casename]

            # self.stats.loc[casename, "initial_dens"] = case.options["Nd+"]["function"] * Nnorm
            self.stats.loc[casename, "line_dens"] = case.avg_density
            self.stats.loc[casename, "target_flux"] = case.target_flux
            self.stats.loc[casename, "target_temp"] = case.target_temp


        self.stats.sort_values(by="line_dens", inplace=True)


    def read_sk(self, path_solkit, timestep, verbose = False):  
        
        mode = platform.system()
        sk = defaultdict(dict)
        
        # Coulomb logarithm for electron-ion collisions:
        def lambda_ei(n, T, T_0, n_0, Z_0):
            if T * T_0 < 10.00 * Z_0 ** 2:
                return 23.00 - np.log(np.sqrt(n * n_0 * 1.00E-6) * Z_0 * (T * T_0) ** (-3.00/2.00))
            else:
                return 24.00 - np.log(np.sqrt(n * n_0 * 1.00E-6) / (T * T_0))

        # def cs_0(gamma, Z, T)
        
        """"""""
        #READING SETTINGS
        """"""""
        
        path_normfile = os.path.join(path_solkit, "INPUT", "NORMALIZATION_INPUT.txt")
        with open(path_normfile) as f:
            normfile = f.readlines()
            
        Z = float(normfile[0].split()[2]) # from NORMALISATION_INPUT
        AA = float(normfile[1].split()[2])
        T_norm = float(normfile[2].split()[2]) # from NORMALISATION_INPUT, in eV
        n_norm = float(normfile[3].split()[2]) # from NORMALISATION_INPUT, in m-3
        path_neut_heat = os.path.join(path_solkit, "INPUT", "NEUT_AND_HEAT_INPUT.txt")
        with open(path_neut_heat) as f:
            neut_heat = f.readlines()
            

        for line in neut_heat:
            if "N_HEATING" in line.split():
                sk["N_HEATING"] = int(line.split()[2])
                
            if "HEAT_POWER" in line.split():
                sk["HEAT_POWER"] = float(line.split()[2])
        
        """"""""
        #CONSTANTS AND NORMALISATION
        """"""""
        
        epsilon_0 = 8.854188e-12 # vaccuum permittivity
        i_mass = 1.67272e-27 * 2
        el_mass = 9.10938e-31
        el_charge = 1.602189e-19
        k_b = 1.3806488e-23
        gamma_ee_0 = el_charge ** 4 / (4 * np.pi * (el_mass * epsilon_0) ** 2) 
        T_J = T_norm * el_charge
        gamma_ei_0 = Z ** 2 * gamma_ee_0
        v_t = np.sqrt(2.0 * T_J / el_mass) # thermal velocity
        t_norm = v_t ** 3 / (gamma_ei_0 * n_norm * lambda_ei(1.0, 1.0, T_norm, n_norm, Z) / Z) # 90deg ei collision time
        x_norm = v_t * t_norm # thermal ei collision mfp

        # Heat flux normalisation from p 99 of Mijin's thesis
        q_norm = el_mass * n_norm * v_t**3
        
        
        list_missing = []
        
        """"""""
        #READING
        """"""""

        if mode == "Windows":
            path_case_sk = str(path_solkit) + "\OUTPUT" 

            sk["Te"] = np.loadtxt(
                path_case_sk + r"\TEMPERATURE\TEMPERATURE_{}.txt".format(timestep)) * T_norm # eV
            sk["Ne"] = np.loadtxt(
                path_case_sk + r"\DENSITY\DENSITY_{}.txt".format(timestep)) * n_norm # in m-3
            sk["Nn"] = np.loadtxt(
                path_case_sk + r"\NEUTRAL_DENS\NEUTRAL_DENS_{}.txt".format(timestep)) * n_norm # in m-3, normalised by n_norm
            sk["x_grid"] = np.loadtxt(
                path_case_sk + r"\GRIDS\X_GRID.txt") * x_norm # in m, normalised by x_norm
            sk["heat_flow_x"] = np.loadtxt(
                path_case_sk + r"\HEAT_FLOW_X\HEAT_FLOW_X_{}.txt".format(timestep)) * q_norm # electron conduction heat flux
            sk["Vi"] = np.loadtxt(
                path_case_sk + r"\ION_VEL\ION_VEL_{}.txt".format(timestep)) * v_t    
        
        # New variables that may not be in all versions of SOLKiT:
        params = defaultdict(dict)
        params["Ti"]["name"] = "ION_TEMPERATURE"
        params["Ti"]["norm"] = T_norm
        params["Tn"] =  {"name": "NEUTRAL_TEMPERATURE", "norm" : T_norm}

        params["Ni"]["name"] = "ION_DENS"
        params["Ni"]["norm"] = n_norm
        params["S_REC"]["name"] = "S_REC"
        params["S_REC"]["norm"] = n_norm / t_norm
        params["S_ION_M"]["name"] = "S_ION_M" # Maxwellian rates. S_ION_SK is kinetic rates
        params["S_ION_M"]["norm"] = n_norm / t_norm * -1
        params["q_cx"]["name"] = "CX_E_RATE"
        params["q_cx"]["norm"] = n_norm * el_charge * T_norm / t_norm
        params["Ve"]["name"] = "FLOW_VEL_X"
        params["Ve"]["norm"] = v_t
        params["Vn"] = {"name": "NEUTRAL_VEL", "norm" : v_t}
        params["Vn_perp"] =  {"name": "NEUTRAL_VEL_PERP", "norm" : v_t}
        
        # params["Riz"]["name"] = "ION_E_RATE"
        # params["Riz"]["norm"] = n_norm * el_charge * T_norm / t_norm
        
        params["deex_e_rate"]["name"] = "DEEX_E_RATE"
        params["deex_e_rate"]["norm"] = n_norm * el_charge * T_norm / t_norm
        
        params["ex_e_rate"]["name"] = "EX_E_RATE"
        params["ex_e_rate"]["norm"] = n_norm * el_charge * T_norm / t_norm
        
        params["rad_deex_e_rate"]["name"] = "RAD_DEEX_E_RATE"
        params["rad_deex_e_rate"]["norm"] = n_norm * el_charge * T_norm / t_norm
        
        params["rad_rec_e_rate"]["name"] = "RAD_REC_E_RATE"
        params["rad_rec_e_rate"]["norm"] = n_norm * el_charge * T_norm / t_norm
        
        params["rec_3b_e_rate"]["name"] = "REC_3B_E_RATE"
        params["rec_3b_e_rate"]["norm"] = n_norm * el_charge * T_norm / t_norm
        
        params["ion_e_rate"]["name"] = "ION_E_RATE"
        params["ion_e_rate"]["norm"] = n_norm * el_charge * T_norm / t_norm
        
        params["ei_e_rate"]["name"] = "EI_E_RATE"
        params["ei_e_rate"]["norm"] = n_norm * el_charge * T_norm / t_norm
        
        params["ION_VEL"]["name"] = "ION_VEL"
        params["ION_VEL"]["norm"] = v_t
        
        params["FLOW_VEL_X"]["name"] = "FLOW_VEL_X"
        params["FLOW_VEL_X"]["norm"] = v_t
        
        params["DENSITY"]["name"] = "DENSITY"
        params["DENSITY"]["norm"] = n_norm
        
        params["NEUTRAL_DENS"]["name"] = "NEUTRAL_DENS"
        params["NEUTRAL_DENS"]["norm"] = n_norm
        
        params["TEMPERATURE"]["name"] = "TEMPERATURE"
        params["TEMPERATURE"]["norm"] = T_norm
        
        params["E_FIELD_X"]["name"] = "E_FIELD_X"
        params["E_FIELD_X"]["norm"] = el_mass * v_t / (t_norm * el_charge)
        
        sk_raw = dict()
    
        for param in params.keys():
            try:
                sk_raw[param] = np.loadtxt(
                        path_case_sk + r"\{}\{}_{}.txt".format(params[param]["name"], params[param]["name"], timestep))

                sk[param] = sk_raw[param] * params[param]["norm"]
                
            except:
                if verbose:
                    print(">> Could not read {} ({})".format(param, params[param]["name"]))
                sk[param] = np.zeros_like(sk["Ne"])
                list_missing.append(param)
            
        """"""""
        # DERIVED VARIABLES
        """"""""
        
        sk["Siz"] = sk["S_ION_M"]
        sk["Srec"] = sk["S_REC"]
        sk["S"] = sk["S_ION_M"] + sk["S_REC"]
        
        # Net excitation
        sk["Rex"] = sk["ex_e_rate"] + sk["rad_deex_e_rate"] - sk["deex_e_rate"]
        sk["Rrec"] = sk["rad_rec_e_rate"] - sk["rec_3b_e_rate"]
        sk["Riz"] = sk["Siz"] * 13.6 * el_charge
        sk["R"] = sk["ex_e_rate"] + sk["ion_e_rate"]  + sk["rad_rec_e_rate"] \
            -sk["deex_e_rate"] - sk["rec_3b_e_rate"] \
        # + sk["rad_deex_e_rate"]
        
        #if "Ti" in list_missing:

        
        # else:
        #     # Thermal energy that plasma receives from a newly ionised neutral
        #     sk["Eiz"] = -(3/2 * sk_raw["Siz"] * 3/T_norm) * n_norm*el_charge*T_norm/t_norm # sk_plotting_functions, Line 1548

        # Note that ion and electron vel are the same in these cases.
        # in m/s, normalised to electron thermal speed
        
        #----------- Cell widths
        x_grid = sk["x_grid"]
        dxc = [None] * len(x_grid)
        dxc[0] = 2 * (x_grid[1] - x_grid[0])
        dxc[-1] = 2 * (x_grid[-1] - x_grid[-2])
        for i in range(1, len(x_grid)-1):
            dxc[i] = x_grid[i+1] - x_grid[i-1]
        sk["dxc"] = np.array(dxc)
        
        # print(sk["x_grid"])
        sk["heating_length"] = sk["x_grid"][ int(2*sk["N_HEATING"]-2) ] + dxc[ int(2*sk["N_HEATING"]-2) ]/2 + dxc[0]/2
        sk["heat_source"] = sk["heating_length"] * sk["HEAT_POWER"]

        
        """"""""
        # CALCULATING SHEATH CONDITIONS
        """"""""
        
        # Calculating positions
        # Add cell edge for target BC
        # Note insert puts value before given pos - last value is achieved with len()
        sk["x_grid"] = np.insert(sk["x_grid"], len(sk["x_grid"]), 
                                                            sk["x_grid"][-1] + (sk["x_grid"][-1] - sk["x_grid"][-2]))

        # Renormalise to 0=upstream BC (but note we don't have a cell there)
        sk["x_grid"] = sk["x_grid"] + sk["x_grid"][1]
        sk["pos"] = sk["x_grid"]

        # Calculate sound speed
        
        if "Ti" not in list_missing:
            sk["Cs"] = np.sqrt(1 * Z  * el_charge * 2 * sk["Te"] / i_mass)
            sk["Cs"] = np.sqrt(2*el_charge*sk["Te"]/i_mass)
        else:
            if verbose:
                print("Apparently this is an ion temperature case")
            sk["Cs"] = np.sqrt(1 * Z  * el_charge * (sk["Ti"] + sk["Te"]) / i_mass) # Gamma is 1 not 5/3 because in 1D you have 1 DOF.
        # k_b needs keV
        

        sk["M"] = sk["Vi"] / sk["Cs"]
        
        # Set all params to NaN at the sheath apart from ones that are defined there
        for param in sk.keys():
            if param not in ["Nn", "Ni", "Ne", "Vi", 
                            "M", "Cs", "Ti", "Te", 
                            "x_grid", "pos", "N_HEATING", 
                            "heating_length","HEAT_POWER",
                            "heat_source"] and param not in list_missing:
                sk[param] = np.insert(sk[param], len(sk[param]), np.nan)

        # Temperature: dT is 0 on each BC, so copy first and last domain temps into there
        sk["Te"] = np.insert(
            sk["Te"], len(sk["Te"]), 
            sk["Te"][-1])
        
        if "Ti" not in list_missing:
            sk["Ti"] = np.insert(
                sk["Ti"], len(sk["Ti"]), 
                sk["Ti"][-1])        

        # Sound speed: same as temperature
        sk["Cs"] = np.insert(
            sk["Cs"], len(sk["Cs"]), 
            sk["Cs"][-1])

        # Mach number: set to 1
        sk["M"] = np.insert(
            sk["M"], len(sk["M"]), 
            1)

        # Velocity: based on M=1
        sk["Vi"] = np.insert(
            sk["Vi"], len(sk["Vi"]), 
            sk["Cs"][-1])

        # Density: the target density scales by the same ratio as the last two cells
        # Upstream density BC not actually calculated in the code, because there's no flux in the BC. 
        sk["Ne"] = np.insert(
            sk["Ne"], len(sk["Ne"]), 
            sk["Ne"][-1] * (sk["Ne"][-1] / sk["Ne"][-2]))
        
        if "Ni" not in list_missing:
            sk["Ni"] = np.insert(
                sk["Ni"], len(sk["Ni"]), 
                sk["Ni"][-1] * (sk["Ni"][-1] / sk["Ni"][-2]))

        # Flattening neutral density profile (obtaining density of ALL neutral species)
        num_states = sk["Nn"].shape[1]
        # Get all of the states out. Nn1 = ground state, Nn2 = n=2 state, etc. Up to 30
        for i in range(num_states):
            state = i + 1
            x = f"Nn{state}"
            sk[x] = sk["Nn"][:,i]
            # sk[x] = np.insert(sk[x], len(x), sk[x][-1] * (sk[x][-1] / sk[x][-2]))
            sk[x] = np.insert(sk[x], len(x), 0)
        
        Nn_array = sk["Nn"]
        sk["Nn"] = np.sum(sk["Nn"], axis = 1)

        # Repeat for neutral density
        # Not actually usd in the code, so this is made up. Used in target in the new version code 
        sk["Nn"] = np.insert(
            sk["Nn"], len(sk["Nn"]), 
            sk["Nn"][-1] * (sk["Nn"][-1] / sk["Nn"][-2]))
        
        # sk["Nn"] = np.insert(
            # sk["Nn"], len(sk["Nn"]), 0)
            

        """"""""
        # DERIVED VARIABLES
        """"""""

        sk["NVi"] = sk["Ne"] * sk["Vi"] # derived variable: plasma flux.
        sk["Ntot"] = sk["Ne"] + sk["Nn"]
        
        if "Ti" not in list_missing:
        # Density weighted average temperature (but density is the same cause of quasineutrality)
            sk["T_mean"] = [np.mean([sk["Te"][x], sk["Ti"][x]]) for x in range(len(sk["Te"]))]
            
        if "Ti" not in list_missing:
            sk["Pe"] = sk["Ne"] * sk["Te"] * el_charge # [m-3] [eV] [J/eV]
            sk["Pi"] = sk["Ni"] * sk["Ti"] * el_charge # [m-3] [eV] [J/eV]
            sk["P"] = sk["Pe"] + sk["Pi"]
        else:
            sk["P"] = sk["Ne"] * sk["Te"] * el_charge # [m-3] [eV] [J/eV]

        if "Vn" not in list_missing:
            sk["NVn"] = sk["Nn"] * sk["Vn"]

        if "Tn" not in list_missing:
            sk["Pn"] = sk["Tn"] * sk["Nn"] * el_charge

            
        for param in list_missing:
            sk[param] = np.insert(sk[param], -1, 0)
            
        sk["t_norm"] = t_norm
        sk["T_norm"] = T_norm
        sk["x_norm"] = x_norm
        sk["v_t_norm"] = v_t
        sk["gamma_ee_0"] = gamma_ee_0
        sk["gamma_ei_0"] = gamma_ei_0
        sk["n_norm"] = n_norm
        sk["Nn_array"] = Nn_array    
        
        if verbose:
            print("SOL-KiT case read: {}".format(path_solkit))
        
        return dict(sk)



