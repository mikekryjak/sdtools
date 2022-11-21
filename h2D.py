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

        if "Ph" in ds.data_vars:
            ds["Th"] = ds["Ph"] / ds["Nh"] / q_e
            ds["Th"].attrs.update({
                "Th": {
                "units": "eV",
                "standard_name": "neutra; temperature (h)",
                "long_name": "Neutral temperature (h)",
                }})

        if "Pd" in ds.data_vars:
            ds["Td"] = ds["Pd"] / ds["Nd"] / q_e
            ds["Td"].attrs.update({
                "Td": {
                "units": "eV",
                "standard_name": "neutral temperature (d)",
                "long_name": "Neutral temperature (d)",
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


    def plot_residuals(self):
        """
        Scaled residual calculation based on ANSYS Fluent
        From 26.13-19 in https://www.afs.enea.it/project/neptunius/docs/fluent/html/ug/node812.htm       
        - Take RMS of rate of change in each parameter over whole domain
        - Normalise by maximum value of this parameter within first 5 iterations
        - Plots parameters corresponding to equations solved (density, momentum, pressure)
        """
        # Find parameters (species dependent)
        list_params = ["Ne", "Pe"]

        for var in self.ds.data_vars:
            if "NV" in var and not any([x in var for x in ["S", ")", "_"]]):
                list_params.append(var)
            if "P" in var and not any([x in var for x in ["S", ")", "_", "e"]]):
                list_params.append(var)
        list_params.sort()

        
        res = dict()

        for param in list_params:

            res[param] = np.diff(self.ds[param], axis = 0) # Rate of change
            res[param] = np.sqrt(res[param]**2) # RMS of rate of change
            res[param] = np.mean(res[param], axis = (1,2))
            res[param] = res[param] / np.max(res[param][:4]) # Normalise by max in first 5 iterations

        fig, ax = plt.subplots(dpi = 100)

        for param in list_params:
            ax.plot(res[param], label = param)
        ax.set_yscale("log")
        ax.grid(which = "major", lw = 1)
        ax.grid(which = "minor", lw = 1, alpha = 0.3)
        ax.legend()
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Normalised residual")
        ax.set_title(f"Residual plot: {self.name}")

    def plot_monitors(self, to_plot):
        """
        Plot time histories of parameters (density, pressure, or momentum)
        In each plot the solid line is the mean and dashed lines 
        represent the min/max at each timestep.
        Momentum is shown as an absolute value
        """

        # Find parameters (species dependent)
        # list_params = ["Ne", "Pe"]
        to_plot = "density"
        list_params = []
        if to_plot == "pressure":
            for var in self.ds.data_vars:
                if "P" in var and not any([x in var for x in ["S", ")", "_", ]]):
                    list_params.append(var)
        elif to_plot == "density":
            for var in self.ds.data_vars:
                if "N" in var and not any([x in var for x in ["S", ")", "_", "V"]]):
                    list_params.append(var)
        elif to_plot == "momentum":
            for var in self.ds.data_vars:
                if "NV" in var and not any([x in var for x in ["S", ")", "_"]]):
                    list_params.append(var)

        list_params.sort()

        data = dict()

        for param in list_params:
            data[param] = dict()
            data[param]["mean"] = np.mean(self.ds[param], axis = (1,2))
            data[param]["max"] = np.max(self.ds[param], axis = (1,2))
            data[param]["min"] = np.min(self.ds[param], axis = (1,2))

            if to_plot == "momentum":
                for key in data[param]:
                    data[param][key] = np.abs(data[param][key])

        colors = ["teal", "darkorange", "firebrick",  "limegreen", "magenta", "cyan", "navy"]
        fig, ax = plt.subplots(dpi = 100)

        for i, param in enumerate(list_params):
            ax.plot(data[param]["mean"], ls = "-", label = f"{param}", color = colors[i])
            ax.plot(data[param]["max"], ls = ":",  color = colors[i])
            ax.plot(data[param]["min"], ls = ":",  color = colors[i])

        ax.set_yscale("log")
        ax.grid(which = "major", lw = 1)
        ax.grid(which = "minor", lw = 1, alpha = 0.3)
        ax.legend(loc = "upper left", bbox_to_anchor=(1,1))
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Normalised residual")
        ax.set_title(f"{to_plot}: {self.name}")



def constants(name):
    
    d = dict()
    d["mass_p"] = 1.6726219e-27 # Proton mass [kg]
    d["mass_e"] = 9.1093837e-31 # Electron mass [kg]
    d["a0"] = 5.29177e-11 # Bohr radius [m]
    d["q_e"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    d["k_b"] = 1.3806488e-23 # Boltzmann self.ant [JK^-1]
    
    return d[name]