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

    def case_2D(casepath, gridfilepath, verbose = False, keep_boundaries = True, squeeze = True):

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

        if squeeze:
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
        self.make_regions()

    

    def unnormalise(self):
        self.calc_norms()
        

        for data_var in self.norms.keys():
            # Unnormalise variables and coordinates
            if data_var in self.ds.variables or data_var in self.ds.coords:
                if data_var not in self.normalised_vars:
                    self.ds[data_var] = self.ds[data_var] * self.norms[data_var]["conversion"]
                    self.normalised_vars.append(data_var)
                self.ds[data_var].attrs.update(self.norms[data_var])

    def derive_vars(self):
        ds = self.ds
        q_e = constants("q_e")

        if "Ph+" in ds.data_vars:
            ds["Th+"] = ds["Ph+"] / (ds["Nh+"] * q_e)
            ds["Th+"].attrs.update({
                "Th+": {
                "units": "eV",
                "standard_name": "ion temperature (h+)",
                "long_name": "Ion temperature (h+)",
                }})

        if "Ph" in ds.data_vars:
            ds["Th"] = ds["Ph"] / (ds["Nh"] * q_e)
            ds["Th"].attrs.update({
                "Th": {
                "units": "eV",
                "standard_name": "neutra; temperature (h)",
                "long_name": "Neutral temperature (h)",
                }})

        if "Pd" in ds.data_vars:
            ds["Td"] = ds["Pd"] / (ds["Nd"] * q_e)
            ds["Td"].attrs.update({
                "Td": {
                "units": "eV",
                "standard_name": "neutral temperature (d)",
                "long_name": "Neutral temperature (d)",
                }})

        if "Pd+" in ds.data_vars:
            ds["Td+"] = ds["Pd+"] / (ds["Nd+"] * q_e)
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

        "dx": {
            "conversion": m["rho_s0"]**2 * m["Bnorm"],
            "units": "Wb",
            "standard_name": "radial cell width",
            "long_name": "Radial cell width in flux space",
        },
        
        "dy": {
            "conversion": 1,
            "units": "radian",
            "standard_name": "poloidal cell angular width",
            "long_name": "Poloidal cell angular width",
        },
        
        "J": {
            "conversion": m["rho_s0"] / m["Bnorm"],
            "units": "m/radianT",
            "standard_name": "Jacobian",
            "long_name": "Jacobian to translate from flux to cylindrical coordinates in real space",
        },
        
        "Th+": {
            "conversion": m["Tnorm"],
            "units": "eV",
            "standard_name": "ion temperature (h+)",
            "long_name": "Ion temperature (h+)",
        },
        
        "Td+": {
            "conversion": m["Tnorm"],
            "units": "eV",
            "standard_name": "ion temperature (d+)",
            "long_name": "Ion temperature (d+)",
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

        "Nd+": {
            "conversion": m["Nnorm"],
            "units": "Pa",
            "standard_name": "ion density (d+)",
            "long_name": "Ion density (d+)"
        },

        "Nd": {
            "conversion": m["Nnorm"],
            "units": "Pa",
            "standard_name": "neutral density (d)",
            "long_name": "Neutral density (d)"
        },

        "Pe": {
            "conversion": m["Nnorm"] * m["Tnorm"] * q_e,
            "units": "Pa",
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

        "Rd+_ex": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "exciation radiation (d+)",
            "long_name": "Multi-step ionisation radiation (d+)"
        },

        "Sd+_iz": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units": "m-3s-1",
            "standard_name": "ionisation",
            "long_name": "Ionisation ion source (d+)"
        },

        
        
        }

        self.norms = d
        
    def slices(self, name):
        """
        DOUBLE NULL ONLY
        Pass this touple to a field of any parameter spanning the grid
        to select points of the appropriate region.
        Each slice is a tuple: (x slice, y slice)
        Use it as: selected_array = array[slice] where slice = (x selection, y selection) = output from this method.
        """

        slices = dict()

        slices["all"] = (slice(None,None), slice(None,None))

        slices["inner_core"] = (slice(0,self.ixseps1), np.r_[slice(self.j1_1g + 1, self.j2_1g+1), slice(self.j1_2g + 1, self.j2_2g + 1)])
        slices["outer_core"] = (slice(self.ixseps1, None), slice(0, self.nyg))

        slices["outer_core_edge"] = (slice(0+self.MXG,1+self.MXG), slice(self.j1_2g + 1, self.j2_2g + 1))
        slices["inner_core_edge"] = (slice(0+self.MXG,1+self.MXG), slice(self.j1_1g + 1, self.j2_1g + 1))

        slices["inner_lower_pfr"] = (slice(0, self.ixseps1), slice(None, self.j1_1g))
        slices["outer_lower_pfr"] = (slice(0, self.ixseps1), slice(self.j2_2g+1, self.nyg))

        slices["lower_pfr"] = (slice(0, self.ixseps1), np.r_[slice(None, self.j1_1g+1), slice(self.j2_2g+1, self.nyg)])
        slices["upper_pfr"] = (slice(0, self.ixseps1), slice(self.j2_1g+1, self.j1_2g+1))

        slices["outer_midplane_a"] = (slice(None, None), int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g)
        slices["outer_midplane_b"] = (slice(None, None), int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g + 1)

        slices["inner_midplane_a"] = (slice(None, None), int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g + 1)
        slices["inner_midplane_b"] = (slice(None, None), int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g)

        return slices[name]


    def make_regions(self):
        """
        Make dataset slices for areas of interest, such as OMP, IMP, targets etc
        """
        data = self.ds
        meta = self.ds.metadata

        # *****OMP*****
        # Find point between j1_2 and j2_2 with highest R coordinate
        # Then interpolate from neighbouring cell centres to the OMP cell boundary
        x = data.isel(t = -1, x = -1, theta = slice(meta["jyseps1_2"], meta["jyseps2_2"]))
        Rmax_id = x.R.values.argmax()
        omp_id_a = meta["jyseps1_2"] + Rmax_id - 1
        omp_id_b = meta["jyseps1_2"] + Rmax_id 

        omp = (data.isel(theta = omp_id_a) + data.isel(theta = omp_id_b)) /2

        # Above operation doesn't allow coordinates to pass through
        for coord in ["R", "Z"]:
            omp.coords[coord] = (data.isel(theta = omp_id_a)[coord] + data.isel(theta = omp_id_b)[coord]) /2

        self.omp = omp


        # *****IMP*****
        # Find point between j1_1 and j1_2 with lowest R coordinate
        # Then interpolate from neighbouring cell centres to the IMP cell boundary
        x = data.isel(t = -1, x = -1, theta = slice(meta["jyseps1_1"], meta["jyseps1_2"]))
        Rmin_id = x.R.values.argmin()
        imp_id_a = meta["jyseps1_1"] + Rmin_id 
        imp_id_b = meta["jyseps1_1"] + Rmin_id + 1

        imp = (data.isel(theta = imp_id_a) + data.isel(theta = imp_id_b)) /2
        for coord in ["R", "Z"]:
            imp.coords[coord] = (data.isel(theta = imp_id_a)[coord] + data.isel(theta = imp_id_b)[coord]) /2

        self.imp = imp

    def summarise_grid(self):
        meta = self.ds.metadata
        print(f' - ixseps1: {meta["ixseps1"]}    // id of first cell after separatrix 1')
        print(f' - ixseps2: {meta["ixseps2"]}    // id of first cell after separatrix 2')
        print(f' - jyseps1_1: {meta["jyseps1_1"]}    // near lower inner')
        print(f' - jyseps1_2: {meta["jyseps1_2"]}    // near lower outer')
        print(f' - jyseps2_1: {meta["jyseps2_1"]}    // near upper outer')
        print(f' - jyseps2_2: {meta["jyseps2_2"]}    // near lower outer')
        print(f' - ny_inner: {meta["ny_inner"]}    // no. poloidal cells in-between divertor regions')
        print(f' - ny: {meta["ny"]}    // total cells in Y (poloidal, does not include guard cells)')
        print(f' - nx: {meta["nx"]}    // total cells in X (radial, includes guard cells)')



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

    def plot_monitors(self, to_plot, ignore = []):
        """
        Plot time histories of parameters (density, pressure, or momentum)
        In each plot the solid line is the mean and dashed lines 
        represent the min/max at each timestep.
        Momentum is shown as an absolute value
        """

        list_params = []
        if to_plot == "pressure":
            for var in self.ds.data_vars:
                if "P" in var and not any([x in var for x in ignore+["S", ")", "_", ]]):
                    list_params.append(var)
        elif to_plot == "density":
            for var in self.ds.data_vars:
                if "N" in var and not any([x in var for x in ignore+["S", ")", "_", "V"]]):
                    list_params.append(var)
        elif to_plot == "momentum":
            for var in self.ds.data_vars:
                if "NV" in var and not any([x in var for x in ignore+["S", ")", "_"]]):
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
        ax.set_ylabel("Value")
        ax.set_title(f"{to_plot}: {self.name}")



def constants(name):
    
    d = dict()
    d["mass_p"] = 1.6726219e-27 # Proton mass [kg]
    d["mass_e"] = 9.1093837e-31 # Electron mass [kg]
    d["a0"] = 5.29177e-11 # Bohr radius [m]
    d["q_e"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    d["k_b"] = 1.3806488e-23 # Boltzmann self.ant [JK^-1]
    
    return d[name]