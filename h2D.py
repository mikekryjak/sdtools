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
        self.extract_geometry()

    

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
        
        "NVd+": {
            "conversion": constants("mass_p") * m["Nnorm"] * m["Cs0"],
            "units": "kgms-1",
            "standard_name": "ion momentum",
            "long_name": "Ion momentum (d+)"
        },
        
        "NVd": {
            "conversion": constants("mass_p") * m["Nnorm"] * m["Cs0"],
            "units": "kgms-1",
            "standard_name": "neutral momentum",
            "long_name": "Neutral momentum (d+)"
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
        slices["core_edge"] = (slice(0+self.MXG,1+self.MXG), np.r_[slice(self.j1_2g + 1, self.j2_2g + 1), slice(self.j1_1g + 1, self.j2_1g + 1)])
        
        slices["outer_sol_edge"] = (slice(-1 - self.MXG,- self.MXG), slice(self.ny_inner+self.MYG*3, self.nyg - self.MYG))
        slices["inner_sol_edge"] = (slice(-1 - self.MXG,- self.MXG), slice(self.MYG, self.ny_inner+self.MYG))
        
        slices["sol_edge"] = (slice(-1 - self.MXG,- self.MXG), np.r_[slice(self.j1_1g + 1, self.j2_1g + 1), slice(self.ny_inner+self.MYG*3, self.nyg - self.MYG)])
        
        
        slices["inner_lower_target"] = (slice(None,None), slice(self.MYG, self.MYG + 1))
        slices["inner_upper_target"] = (slice(None,None), slice(self.ny_inner+self.MYG -1, self.ny_inner+self.MYG))
        slices["outer_upper_target"] = (slice(None,None), slice(self.ny_inner+self.MYG*3, self.ny_inner+self.MYG*3+1))
        slices["outer_lower_target"] = (slice(None,None), slice(self.nyg-self.MYG-1, self.nyg - self.MYG))
        
        slices["inner_lower_target_guard"] = (slice(None,None), slice(self.MYG -1, self.MYG))
        slices["inner_upper_target_guard"] = (slice(None,None), slice(self.ny_inner+self.MYG , self.ny_inner+self.MYG+1))
        slices["outer_upper_target_guard"] = (slice(None,None), slice(self.ny_inner+self.MYG*3-1, self.ny_inner+self.MYG*3))
        slices["outer_lower_target_guard"] = (slice(None,None), slice(self.nyg-self.MYG, self.nyg - self.MYG+1))
        
        slices["inner_lower_pfr"] = (slice(0, self.ixseps1), slice(None, self.j1_1g))
        slices["outer_lower_pfr"] = (slice(0, self.ixseps1), slice(self.j2_2g+1, self.nyg))

        slices["lower_pfr"] = (slice(0, self.ixseps1), np.r_[slice(None, self.j1_1g+1), slice(self.j2_2g+1, self.nyg)])
        slices["upper_pfr"] = (slice(0, self.ixseps1), slice(self.j2_1g+1, self.j1_2g+1))
        slices["pfr"] = (slice(0, self.ixseps1), np.r_[ 
                                                       np.r_[slice(None, self.j1_1g+1), slice(self.j2_2g+1, self.nyg)], 
                                                       slice(self.j2_1g+1, self.j1_2g+1)])
        
        slices["lower_pfr_edge"] = (slice(self.MXG, self.MXG+1), np.r_[slice(None, self.j1_1g+1), slice(self.j2_2g+1, self.nyg)])
        slices["upper_pfr_edge"] = (slice(self.MXG, self.MXG+1), slice(self.j2_1g+1, self.j1_2g+1))
        slices["pfr_edge"] = (slice(self.MXG, self.MXG+1), np.r_[
                                                                    np.r_[slice(None, self.j1_1g+1), slice(self.j2_2g+1, self.nyg)],
                                                                    slice(self.j2_1g+1, self.j1_2g+1)])
        
        slices["outer_midplane_a"] = (slice(None, None), int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g)
        slices["outer_midplane_b"] = (slice(None, None), int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g + 1)

        slices["inner_midplane_a"] = (slice(None, None), int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g + 1)
        slices["inner_midplane_b"] = (slice(None, None), int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g)

        return slices[name]


    def extract_geometry(self):
        """
        Perpare geometry variables
        """
        data = self.ds
        meta = self.ds.metadata

        # self.Rxy = meta["Rxy"]    # R coordinate array
        # self.Zxy = meta["Zxy"]    # Z coordinate array
        
        self.ixseps1 = meta["ixseps1"]
        self.MYG = meta["MYG"]
        self.MXG = 2
        self.ny_inner = meta["ny_inner"]
        self.ny = meta["ny"]
        self.nyg = self.ny + self.MYG * 4 # with guard cells
        self.nx = meta["nx"]

        self.j1_1 = meta["jyseps1_1"]
        self.j1_2 = meta["jyseps1_2"]
        self.j2_1 = meta["jyseps2_1"]
        self.j2_2 = meta["jyseps2_2"]

        self.j1_1g = self.j1_1 + self.MYG
        self.j1_2g = self.j1_2 + self.MYG * 3
        self.j2_1g = self.j2_1 + self.MYG
        self.j2_2g = self.j2_2 + self.MYG * 3
        
        # Cell areas in flux space
        # dV = dx * dy * dz * J where dz is assumed to be 2pi in 2D
        self.dx = data["dx"]
        self.dy = data["dy"]
        self.dydx = data["dy"] * data["dx"]    # Poloidal surface area
        self.J = data["J"]
        dz = 2*np.pi    # Axisymmetric
        self.dv = self.dydx * dz * data["J"]    # Cell volume
        
        # Cell areas in real space
        # TODO: Check these against dx/dy to ensure volume is the same
        # dV = (hthe/Bpol) * (R*Bpol*dr) * dy*2pi = hthe * dy * dr * 2pi * R
        self.dr = self.dx / (self.ds.R * self.ds.Bpxy)    # Length of cell in radial direction
        self.hthe = self.J * self.ds["Bpxy"]    # h_theta
        self.dl = self.dy * self.hthe    # poloidal arc length
        
    def collect_boundaries(self):
        self.boundaries = dict()
        for target_name in ["inner_lower_target", "inner_upper_target", "outer_upper_target", "outer_lower_target"]:
            self.boundaries[target_name] = Target(self, target_name)
            

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
        
    def plot_ddt(self, smoothing = 50, volume_weighted = True):
        """
        RMS of all the ddt parameters, which are convergence metrics.
        Inputs:
        smoothing: moving average period used for plot smoothing (default = 20. 1 is no smoothing)
        volume_weighted: weigh the ddt by cell volume
        """
        # Find parameters (species dependent)
        list_params = []

        for var in self.ds.data_vars:
            if "ddt" in var and not any([x in var for x in []]):
                list_params.append(var)
        list_params.sort()
        
        # Account for case if not enough timesteps for smoothing
        if len(self.ds.coords["t"]) < smoothing:
            smoothing = len(self.ds.coords) / 10

        res = dict()
        ma = dict()

        for param in list_params:

            if volume_weighted:
                res[param] = (self.ds[param] * self.dv) / np.sum(self.dv)    # Cell volume weighted
            else:
                res[param] = self.ds[param]
            res[param] = np.sqrt(np.mean(res[param]**2, axis = (1,2)))    # Root mean square
            res[param] = np.convolve(res[param], np.ones(smoothing), "same")    # Moving average with window = smoothing

        fig, ax = plt.subplots(figsize = (8,6), dpi = 100)

        for param in list_params:
            ax.plot(res[param], label = param, lw = 1)
            
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


class Target():
    def __init__(self, case, target_name):
        self.case = case
        data = case.ds
        mass_i = constants("mass_p") * 2

        data["dr"] = data["dx"] / (data["R"] * data["Bpxy"])
        self.last = data.isel(t=-1, x = case.slices(f"{target_name}")[0], theta = case.slices(f"{target_name}")[1])
        self.guard = data.isel(t=-1, x = case.slices(f"{target_name}_guard")[0], theta = case.slices(f"{target_name}_guard")[1])

        def bndry_val(param):
            return (self.last[param].values + self.guard[param].values)/2

        try:
            gamma_i = self.ds.options["sheath_boundary_simple"]["gamma_i"]
        except:
            gamma_i = 3.5
            
        try:
            gamma_e = self.ds.options["sheath_boundary_simple"]["gamma_e"]
        except:
            gamma_e = 3.5
            
        self.particle_flux = abs(bndry_val("NVd+")) / mass_i    
        self.heat_flux = gamma_i * bndry_val("Td+") * constants("q_e") * self.particle_flux * 1e-6    # MW
        self.r = np.cumsum(bndry_val("dr"))    # Length along divertor
        # width = np.insert(0,0,width)

        self.total_heat_flux = np.trapz(x = self.r, y = self.heat_flux.squeeze()) * 2*np.pi
        self.total_particle_flux = np.trapz(x = self.r, y = self.particle_flux.squeeze()) * 2*np.pi
        
    def plot(self, what):
        
        fig, ax = plt.subplots(figsize = (6,4), dpi = 100)
        ax.set_xlabel("Radial distance [m]")
        ax.grid()
        
        
        if what == "heat_flux":
            
            ax.set_title(f"{self.case.name}: total heat flux integral: {self.total_heat_flux:,.3f}[MW]")
            ax.plot(self.r, self.heat_flux, c = "k", marker = "o")
            ax.set_ylabel("Target heat flux [MW/m2]")


def constants(name):
    
    d = dict()
    d["mass_p"] = 1.6726219e-27 # Proton mass [kg]
    d["mass_e"] = 9.1093837e-31 # Electron mass [kg]
    d["a0"] = 5.29177e-11 # Bohr radius [m]
    d["q_e"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    d["k_b"] = 1.3806488e-23 # Boltzmann self.ant [JK^-1]
    
    return d[name]