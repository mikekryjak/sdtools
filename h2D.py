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

        self.colors = ["cyan", "lime", "crimson", "magenta", "black", "red"]

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

        def custom_core_ring(i):
            """
            Creates custom SOL ring slice within the core.
            i = 0 is at first domain cell.
            i = -2 is at first inner guard cell.
            i = ixseps - MXG is the separatrix.
            """
            if i > self.ixseps1 - self.MXG:
                raise Exception("i is too large!")
            
            return (slice(0+self.MXG+i,1+self.MXG+i), np.r_[slice(self.j1_2g + 1, self.j2_2g + 1), slice(self.j1_1g + 1, self.j2_1g + 1)])
            
        def custom_sol_ring(i, region  = "all"):
            """
            Creates custom SOL ring slice beyond the separatrix.
            i = index of SOL ring (0 is separatrix, 1 is first SOL ring)
            region = all, inner, inner_lower, inner_upper, outer, outer_lower, outer_upper
            """
            
            i = i + self.ixseps1 - 1
            if i > self.nx - self.MXG*2 :
                raise Exception("i is too large!")
            
            if region == "all":
                return (slice(i+1,i+2), np.r_[slice(0+self.MYG, self.j2_2g + 1), slice(self.j1_1g + 1, self.nyg - self.MYG)])
            
            if region == "inner":
                return (slice(i+1,i+2), slice(0+self.MYG, self.ny_inner + self.MYG))
            if region == "inner_lower":
                return (slice(i+1,i+2), slice(0+self.MYG, int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g +2))
            if region == "inner_upper":
                return (slice(i+1,i+2), slice(int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g, self.ny_inner + self.MYG))
            
            if region == "outer":
                return (slice(i+1,i+2), slice(self.ny_inner + self.MYG*3, self.nyg - self.MYG))
            if region == "outer_lower":
                return (slice(i+1,i+2), slice(int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g, self.nyg - self.MYG))
            if region == "outer_upper":
                return (slice(i+1,i+2), slice(self.ny_inner + self.MYG*3, int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g + 2))

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
        
        slices["custom_core_ring"] = custom_core_ring
        slices["custom_sol_ring"] = custom_sol_ring
        
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
    
    def plot_slice(self, slicer, dpi = 100):
        """
        Indicates region of cells in X, Y and R, Z space for implementing Hermes-3 sources
        You must provide a slice() object for the X and Y dimensions which is a tuple in the form (X,Y)
        X is the radial coordinate (excl guards) and Y the poloidal coordinate (incl guards)
        WARNING: only developed for a connected double null. Someone can adapt this to a single null or DDN.
        """
        
        meta = self.ds.metadata
        xslice = slicer[0]
        yslice = slicer[1]

        # Region boundaries
        ny = meta["ny"]     # Total ny cells (incl guard cells)
        nx = meta["nx"]     # Total nx cells (excl guard cells)
        Rxy = self.ds["R"].values    # R coordinate array
        Zxy = self.ds["Z"].values    # Z coordinate array
        MYG = meta["MYG"]

        # Array of radial (x) indices and of poloidal (y) indices in the style of Rxy, Zxy
        x_idx = np.array([np.array(range(nx))] * int(ny + MYG * 4)).transpose()
        y_idx = np.array([np.array(range(ny + MYG*4))] * int(nx))

        # Slice the X, Y and R, Z arrays and vectorise them for plotting
        xselect = x_idx[xslice,yslice].flatten()
        yselect = y_idx[xslice,yslice].flatten()
        rselect = Rxy[xslice,yslice].flatten()
        zselect = Zxy[xslice,yslice].flatten()

        # Plot
        fig, axes = plt.subplots(1,3, figsize=(12,5), dpi = dpi, gridspec_kw={'width_ratios': [2.5, 1, 2]})
        fig.subplots_adjust(wspace=0.3)

        self.plot_xy_grid(axes[0])
        axes[0].scatter(yselect, xselect, s = 4, c = "red", marker = "s", edgecolors = "darkorange", linewidths = 0.5)

        self.plot_rz_grid(axes[1])
        axes[1].scatter(rselect, zselect, s = 20, c = "red", marker = "s", edgecolors = "darkorange", linewidths = 1, zorder = 100)

        self.plot_rz_grid(axes[2], ylim=(-1,-0.25))
        axes[2].scatter(rselect, zselect, s = 20, c = "red", marker = "s", edgecolors = "darkorange", linewidths = 1, zorder = 100)
    
    def plot_xy_grid(self, ax):
        ax.set_title("X, Y index space")
        ax.scatter(self.yflat, self.xflat, s = 1, c = "grey")
        ax.plot([self.yflat[self.j1_1g]]*np.ones_like(self.xflat), self.xflat, label = "j1_1g",   color = self.colors[0])
        ax.plot([self.yflat[self.j1_2g]]*np.ones_like(self.xflat), self.xflat, label = "j1_2g", color = self.colors[1])
        ax.plot([self.yflat[self.j2_1g]]*np.ones_like(self.xflat), self.xflat, label = "j2_1g",   color = self.colors[2])
        ax.plot([self.yflat[self.j2_2g]]*np.ones_like(self.xflat), self.xflat, label = "j2_2g", color = self.colors[3])
        ax.plot(self.yflat, [self.yflat[self.ixseps1]]*np.ones_like(self.yflat), label = "ixseps1", color = self.colors[4])
        ax.plot(self.yflat, [self.yflat[self.ixseps2]]*np.ones_like(self.yflat), label = "ixseps1", color = self.colors[5], ls=":")
        ax.legend(loc = "upper center", bbox_to_anchor = (0.5,-0.1), ncol = 3)
        ax.set_xlabel("Y index (incl. guards)")
        ax.set_ylabel("X index (excl. guards)")

    def plot_rz_grid(self, ax, xlim = (None,None), ylim = (None,None)):
        ax.set_title("R, Z space")
        ax.scatter(self.rflat, self.zflat, s = 0.1, c = "black")
        ax.set_axisbelow(True)
        ax.grid()
        ax.plot(self.Rxy[:,self.j1_1g], self.Zxy[:,self.j1_1g], label = "j1_1g",     color = self.colors[0], alpha = 0.7)
        ax.plot(self.Rxy[:,self.j1_2g], self.Zxy[:,self.j1_2g], label = "j1_2g", color = self.colors[1], alpha = 0.7)
        ax.plot(self.Rxy[:,self.j2_1g], self.Zxy[:,self.j2_1g], label = "j2_1g",     color = self.colors[2], alpha = 0.7)
        ax.plot(self.Rxy[:,self.j2_2g], self.Zxy[:,self.j2_2g], label = "j2_2g", color = self.colors[3], alpha = 0.7)
        ax.plot(self.Rxy[self.ixseps1,:], self.Zxy[self.ixseps1,:], label = "ixseps1", color = self.colors[4], alpha = 0.7, lw = 2)
        ax.plot(self.Rxy[self.ixseps2,:], self.Zxy[self.ixseps2,:], label = "ixseps2", color = self.colors[5], alpha = 0.7, lw = 2, ls=":")

        if xlim != (None,None):
            ax.set_xlim(xlim)
        if ylim != (None,None):
            ax.set_ylim(ylim)
    



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
        
        # Array of radial (x) indices and of poloidal (y) indices in the style of Rxy, Zxy
        self.x_idx = np.array([np.array(range(self.nx))] * int(self.nyg)).transpose()
        self.y_idx = np.array([np.array(range(self.nyg))] * int(self.nx))
        
        self.yflat = self.y_idx.flatten()
        self.xflat = self.x_idx.flatten()
        self.rflat = self.ds.coords["R"].values.flatten()
        self.zflat = self.ds.coords["Z"].values.flatten()

        self.j1_1 = meta["jyseps1_1"]
        self.j1_2 = meta["jyseps1_2"]
        self.j2_1 = meta["jyseps2_1"]
        self.j2_2 = meta["jyseps2_2"]
        self.ixseps2 = meta["ixseps2"]
        self.ixseps1 = meta["ixseps1"]
        self.Rxy = self.ds.coords["R"]
        self.Zxy = self.ds.coords["Z"]

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
        self.hthe = self.J * self.ds["Bpxy"]    # poloidal arc length per radian
        self.dl = self.dy * self.hthe    # poloidal arc length
        
        self.ds["dr"] = self.ds.dx / (self.ds.R * self.ds.Bpxy)
        self.ds["dr"].attrs.update({
            "conversion" : 1,
            "units" : "m",
            "standard_name" : "radial length",
            "long_name" : "Length of cell in the radial direction"})
        
        self.ds["hthe"] = self.ds.J * self.ds["Bpxy"]    # h_theta
        self.ds["hthe"].attrs.update({
            "conversion" : 1,
            "units" : "m/radian",
            "standard_name" : "h_theta: poloidal arc length per radian",
            "long_name" : "h_theta: poloidal arc length per radian"})
        
        self.ds["dl"] = self.ds.dy * self.ds["hthe"]    # poloidal arc length
        self.ds["dl"].attrs.update({
            "conversion" : 1,
            "units" : "m",
            "standard_name" : "poloidal arc length",
            "long_name" : "Poloidal arc length"})
        
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
        ax.legend(loc = "upper left", bbox_to_anchor=(1,1))
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Normalised residual")
        ax.set_title(f"Residual plot: {self.name}")

    def plot_monitors(self, to_plot, what = ["mean", "max", "min"], ignore = []):
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
                    
        else:
            list_params = to_plot

        list_params.sort()
        

        data = dict()

        for param in list_params:
            data[param] = dict()
            if "mean" in what:
                data[param]["mean"] = np.mean(self.ds[param], axis = (1,2))
            if "max" in what:
                data[param]["max"] = np.max(self.ds[param], axis = (1,2))
            if "min" in what:
                data[param]["min"] = np.min(self.ds[param], axis = (1,2))

            if to_plot == "momentum":
                for key in data[param]:
                    data[param][key] = np.abs(data[param][key])

        colors = ["teal", "darkorange", "firebrick",  "limegreen", "magenta", "cyan", "navy"]
        fig, ax = plt.subplots(dpi = 100)

        for i, param in enumerate(list_params):
            if "mean" in what:
                ax.plot(data[param]["mean"], ls = "-", label = f"{param}", color = colors[i])
            if "max" in what:
                ax.plot(data[param]["max"], ls = ":",  color = colors[i])
            if "min" in what:
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
            
class CoreRing():
    """
    Object defining a SOL ring within the core
    For the purposes of calculating fluxes through it
    
    Returns dictionaries of xarray objects per species
    with history of the following parameters
    
    particle_flux
    convective_heat_flux
    diffusive_heat_flux
    total_heat_flux
    ring_temperature
    ring_density
    """
    def __init__(self, case, ring_index = 0):
        self.ds = case.ds
        self.case = case
        self.ring_index = ring_index

        # Hardcoded species list
        self.list_species = ["d+", "e"]
        
        # Hardcoded anomalous coefficients
        self.D = {"d+" : 0.3, "e" : 0.3}
        self.Chi = {"d+" : 0.45, "e" : 0.45}
        
        self.extract_rings()
        self.calculate_fluxes()
        self.sum_fluxes()
        self.update_attributes()

    def extract_rings(self):
        """
        Slice the dataset into the two rings between which we are 
        calculating flux. Also calculate geometry properties
        """
        # Define ring for flux calculation
        self.a_slice = self.case.slices("custom_core_ring")(self.ring_index)    # First ring
        self.b_slice = self.case.slices("custom_core_ring")(self.ring_index+1)    # Second ring

        # Datasets for each ring
        self.a = self.ds.isel(t = -1, x = self.a_slice[0], theta = self.a_slice[1])
        self.b = self.ds.isel(t = -1, x = self.b_slice[0], theta = self.b_slice[1])

        # Geometry properties
        self.dr = self.a["dr"].values/2 + self.b["dr"].values/2    # Distance between cell centres of rings 
        self.R = (self.a["R"].values + self.b["R"].values)/2    # Major radius of the edge in-between rings 
        self.A = (self.a["dl"].values + self.b["dl"].values)/2 * 2*np.pi*self.R    # Surface area of the edge in-between rings
        self.diff = self.ds.diff(dim = "x")    # Difference the entire dataset for dN/dT calculations
        
    def calculate_fluxes(self):
        """
        Calculate heat and particle fluxes
        """

        dN = dict()
        dT = dict()
        grad_N = dict()
        grad_T = dict()
        self.particle_flux = dict()
        self.convective_heat_flux = dict()
        self.diffusive_heat_flux = dict()
        self.total_heat_flux = dict()
        self.ring_temperature = dict()
        self.ring_density = dict()
        
        for species in self.list_species:

            # Use xarray's diff to preserve dimensions when doing a difference, then slice to extract ring
            dN[species] = self.ds[f"N{species}"].diff("x").isel(x = self.a_slice[0], theta = self.a_slice[1])    
            dT[species] = self.ds[f"T{species}"].diff("x").isel(x = self.a_slice[0], theta = self.a_slice[1])    
            grad_N[species] = dN[species] / self.dr
            grad_T[species] = dT[species] / self.dr

            # At edge in-between rings:
            self.ring_temperature[species] = (self.a[f"T{species}"].values + self.b[f"T{species}"].values) / 2     # Temperature [eV]
            self.ring_density[species] = (self.a[f"N{species}"].values + self.b[f"N{species}"].values) / 2     # Density [m-3]

            # Calculate flux (D*-dN/dx) in each cell and multiply by its surface area, then sum along the ring
            self.particle_flux[species] = ((self.D[species] * - grad_N[species]) * self.A).sum("theta")    # s-1

            # Convective: D*-dN/dx * T
            # Diffusive: Chi*-dT/dx * N
            self.convective_heat_flux[species] = ((self.D[species] * - grad_N[species]) * self.A * self.ring_temperature[species]).sum("theta") * 3/2 * constants("q_e") * 1e-6    # Heat flux [MW]
            self.diffusive_heat_flux[species] = ((self.Chi[species] * - grad_T[species]) * self.A * self.ring_density[species]).sum("theta") * 3/2 * constants("q_e") * 1e-6    # Heat flux [MW]
            self.total_heat_flux[species] = self.convective_heat_flux[species] + self.diffusive_heat_flux[species]
        
    def update_attributes(self):

        for species in self.list_species:
            self.particle_flux[species].attrs.update({
                "conversion":1,
                "units":"s-1",
                "standard_name":f"{species} particle flux",
                "long_name":f"{species} particle flux",
            })

            self.convective_heat_flux[species].attrs.update({
                "conversion":1,
                "units":"MW",
                "standard_name":f"{species} convective heat flux",
                "long_name":f"{species} convective heat flux",
            })

            self.diffusive_heat_flux[species].attrs.update({
                "conversion":1,
                "units":"MW",
                "standard_name":f"{species} diffusive heat flux",
                "long_name":f"{species} diffusive heat flux",
            })

            self.total_heat_flux[species].attrs.update({
                "conversion":1,
                "units":"MW",
                "standard_name":f"{species} total heat flux",
                "long_name":f"{species} total heat flux (convective + diffusive)",
            })
            
            self.total_heat_flux_all.attrs.update({
                "standard_name":f"Total plasma heat flux",
                "long_name":f"Total plasma heat flux (convective + diffusive)",
            })
            self.particle_flux_all.attrs.update({
                    "standard_name":f"Particle flux",
                    "long_name":f"Plasma particle flux",
                })
            
    def sum_fluxes(self):
        
        first_species = self.list_species[0]
        
        self.total_heat_flux_all = self.total_heat_flux[first_species].copy()
        self.particle_flux_all = self.particle_flux[first_species].copy()
        
        list_species = self.list_species.copy()
        list_species.remove(first_species)
        
        for species in list_species:
            self.total_heat_flux_all += self.total_heat_flux[species]
            self.particle_flux_all += self.particle_flux[species]
            
        
            
    def plot_location(self):
        self.case.plot_slice(self.case.slices("custom_core_ring")(self.ring_index))
        
    def plot_particle_flux_history(self, dpi = 100):
        fig, ax = plt.subplots(figsize = (5,4), dpi = dpi)
        
        
        for species in self.list_species:
            self.particle_flux[species].plot(ax = ax, label = species)
            
        ax.grid()
        ax.set_ylabel("Particle flux [s-1]")
        ax.set_title(f"Flux for domain core ring {self.ring_index}")
        ax.legend()
        
    def plot_heat_flux_history(self, dpi = 100):
        fig, ax = plt.subplots(figsize = (5,4), dpi = dpi)
        
        
        for species in self.list_species:
            self.total_heat_flux[species].plot(ax = ax, label = species)
            
        ax.grid()
        ax.set_ylabel("Heat flux [MW]")
        ax.set_title(f"Total heat flux for domain core ring {self.ring_index}")
        ax.legend()

    


def constants(name):
    
    d = dict()
    d["mass_p"] = 1.6726219e-27 # Proton mass [kg]
    d["mass_e"] = 9.1093837e-31 # Electron mass [kg]
    d["a0"] = 5.29177e-11 # Bohr radius [m]
    d["q_e"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    d["k_b"] = 1.3806488e-23 # Boltzmann self.ant [JK^-1]
    
    return d[name]