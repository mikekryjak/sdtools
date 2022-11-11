#!/usr/bin/env python3

from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import traceback
import platform
import colorcet as cc
from scipy import stats
from boututils.datafile import DataFile
from boutdata.collect import collect
from boutdata.data import BoutData
import xbout


class Case:

    def __init__(self, casepath, gridfilepath, verbose = False, load = True, process = True):
        self.casename = casepath.split(os.path.sep)[-1]
        self.gridfilepath = gridfilepath
        self.casepath = casepath
        self.datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
        self.inputfilepath = os.path.join(casepath, "BOUT.inp")
        self.gridfilepath = gridfilepath
        self.verbose = verbose
        self.normalised_vars = []

        print(gridfilepath)
        print(casepath)

        if load:
            self.load_dataset()
        if process:
            self.unnormalise()
            self.derive_vars()

    def load_dataset(self):
        self.ds = xbout.load.open_boutdataset(
                datapath = self.datapath, 
                inputfilepath = self.inputfilepath, 
                gridfilepath = self.gridfilepath,
                info = False,
                geometry = "toroidal",
                keep_xboundaries=True,
                keep_yboundaries=True,
                )

        self.ds = self.ds.squeeze(drop = True)

    def unnormalise(self):
        self.calc_norms()
        

        for data_var in self.norms.keys():
            # Unnormalise variables and coordinates
            if data_var in self.ds.variables or data_var in self.ds.coords:
                if data_var not in self.normalised_vars:
                    # print("data_var")
                    # print(self.normalised_vars)
                    self.ds[data_var] = self.ds[data_var] * self.norms[data_var]["conversion"]
                    self.normalised_vars.append(data_var)
                self.ds[data_var].attrs.update(self.norms[data_var])

    def derive_vars(self):
        self.calc_derivations()
        self.derived_vars = []

        for data_var in self.derivations.keys():
            self.ds[data_var] = self.derivations[data_var]["calculation"]
            self.ds[data_var].attrs.update(self.norms[data_var])
            self.derived_vars.append(data_var)

    def calc_derivations(self):
        ds = self.ds
        q_e = constants("q_e")
        d = {

        "Th+": {
            "calculation": ds["Ph+"] / ds["Nh+"] / q_e,
            "units": "eV",
            "standard_name": "ion temperature (h+)",
            "long_name": "Ion temperature (h+)",
        },
            
        }

        self.derivations = d


    def calc_norms(self):
        
        m = self.ds.metadata
        q_e = constants("q_e")
        d = {

        "Th+": {
            "conversion": m["Tnorm"],
            "units": "eV",
            "standard_name": "ion temperature (h+)",
            "long_name": "Ion temperature (h+)",
        },

        "Te": {
            "conversion": m["Tnorm"],
            "units": "eV",
            "standard_name": "electron temperature",
            "long_name": "Electron temperature",
        },

        "Ne": {
            "conversion": m["Nnorm"],
            "units": "m-3",
            "standard_name": "density",
            "long_name": "Electron density"
        },

        "Nh+": {
            "conversion": m["Nnorm"],
            "units": "Pa",
            "standard_name": "ion density (h+)",
            "long_name": "Ion density (h+)"
        },

        "Pe": {
            "conversion": m["Nnorm"] * m["Tnorm"] * q_e,
            "units": "Pe",
            "standard_name": "electron pressure",
            "long_name": "Electron pressure"
        },

        "Ph+": {
            "conversion": m["Nnorm"] * m["Tnorm"] * q_e,
            "units": "Pa",
            "standard_name": "ion pressure (h+)",
            "long_name": "Ion pressure (h+)"
        },
        
        }

        self.norms = d

    

    def diagnose_cvode(self, lims = (0,0), scale = "log"):
        ds = self.ds

        fig, axes = plt.subplots(2,2, figsize = (8,6))

        ax = axes[0,0];  ax.set_yscale(scale)
        ax.plot(ds.coords["t"], ds.data_vars["cvode_nsteps"].values, label = "nsteps")
        ax.plot(ds.coords["t"], ds.data_vars["cvode_nfevals"].values, label = "nfevals")
        ax.plot(ds.coords["t"], ds.data_vars["cvode_npevals"].values, label = "npevals")
        ax.plot(ds.coords["t"], ds.data_vars["cvode_nliters"].values, label = "nliters")

        ax = axes[0,1]
        ax.plot(ds.coords["t"], ds.data_vars["cvode_last_order"].values, label = "last_order", lw = 1)

        ax = axes[1,0]; ax.set_yscale(scale)
        ax.plot(ds.coords["t"], ds.data_vars["cvode_num_fails"].values, label = "num_fails")
        ax.plot(ds.coords["t"], ds.data_vars["cvode_nonlin_fails"].values, label = "nonlin_fails")

        ax = axes[1,1]; ax.set_yscale(scale)
        ax.plot(ds.coords["t"], ds.data_vars["cvode_stab_lims"].values, label = "stab_lims")
        ax.plot(ds.coords["t"], ds.data_vars["cvode_stab_lims"].values, label = "last_step")

        for i in range(2):
            for j in range(2): 
                axes[i,j].grid()
                axes[i,j].legend()
                if lims != (0,0):
                    axes[i,j].set_xlim(lims)


def constants(name):
    
    d = dict()
    d["mass_p"] = 1.6726219e-27 # Proton mass [kg]
    d["mass_e"] = 9.1093837e-31 # Electron mass [kg]
    d["a0"] = 5.29177e-11 # Bohr radius [m]
    d["q_e"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    d["k_b"] = 1.3806488e-23 # Boltzmann self.ant [JK^-1]
    
    return d[name]