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


class Load:
    def __init__(self):
        pass

    def case_1D(casepath):
        datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
        inputfilepath = os.path.join(casepath, "BOUT.inp")

        ds = xbout.load.open_boutdataset(
                datapath = datapath, 
                inputfilepath = inputfilepath, 
                info = False,
                keep_yboundaries=True,
                )

        ds = ds.squeeze(drop = True)

        return Case(ds, casepath)

    def case_2D(casepath, gridfilepath, verbose = False, keep_boundaries = True):
        datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
        inputfilepath = os.path.join(casepath, "BOUT.inp")
        

        print(gridfilepath)
        print(casepath)

        ds = xbout.load.open_boutdataset(
                datapath = datapath, 
                inputfilepath = inputfilepath, 
                gridfilepath = gridfilepath,
                info = False,
                geometry = "toroidal",
                keep_xboundaries=keep_boundaries,
                keep_yboundaries=keep_boundaries,
                )

        ds = ds.squeeze(drop = True)
        return Case(ds, casepath)


class Case:

    def __init__(self, ds, casepath):

        self.ds = ds
        self.name = os.path.split(casepath)[-1]
        self.datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
        self.inputfilepath = os.path.join(casepath, "BOUT.inp")
        self.normalised_vars = []

        if "x" in self.ds.dims:
            self.is_2d = True
        else:
            self.is_2d = False


        self.unnormalise()
        self.derive_vars()

    

    def unnormalise(self):
        self.calc_norms()
        

        for data_var in self.norms.keys():
            # Unnormalise variables and coordinates
            if data_var in self.ds.variables or data_var in self.ds.coords:
                if data_var not in self.normalised_vars:
                    self.ds[data_var] = self.ds[data_var] * self.norms[data_var]["conversion"]
                    self.normalised_vars.append(data_var)
                self.ds[data_var].attrs.update(self.norms[data_var])

    # def derive_vars(self):
    #     self.calc_derivations()
    #     self.derived_vars = []

    #     for data_var in self.derivations.keys():
    #         self.ds[data_var] = self.derivations[data_var]["calculation"]
    #         self.ds[data_var].attrs.update(self.norms[data_var])
    #         self.derived_vars.append(data_var)

    def derive_vars(self):
        ds = self.ds
        q_e = constants("q_e")

        if "Ph+" in ds.data_vars:
            ds["Th+"] = ds["Ph+"] / ds["Nh+"] / q_e
            ds["Th+"].attrs.update({
                "Th+": {
                "units": "eV",
                "standard_name": "ion temperature (h+)",
                "long_name": "Ion temperature (h+)",
                }})

        if "Pd+" in ds.data_vars:
            ds["Td+"] = ds["Pd+"] / ds["Nd+"] / q_e
            ds["Td+"].attrs.update({
                "Td+": {
                "units": "eV",
                "standard_name": "ion temperature (d+)",
                "long_name": "Ion temperature (d+)",
                }})

    def guard_replace(self):

        if self.is_2d == False:
            for data_var in self.ds.data_vars:
                if "x" in self.ds[data_var].dims:
                    pass

        else:
            print("2D guard replacement not done yet")


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

        "Pd+": {
            "conversion": m["Nnorm"] * m["Tnorm"] * q_e,
            "units": "Pa",
            "standard_name": "ion pressure (d+)",
            "long_name": "Ion pressure (d+)"
        },

        "Pd+_src": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "ion energy source (d+)",
            "long_name": "Ion energy source (d+)"
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