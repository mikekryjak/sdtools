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

from hermes3.utils import *
from hermes3.named_selections import *
from hermes3.plotting import *


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

    def case_2D(casepath, 
                gridfilepath = None, 
                verbose = False, 
                keep_boundaries = False, 
                squeeze = True, 
                double_load = False):
        """ 
        Double load returns a case with and without guards.
        """

        datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
        inputfilepath = os.path.join(casepath, "BOUT.inp")
        
        ds = xbout.load.open_boutdataset(
                datapath = datapath, 
                inputfilepath = inputfilepath, 
                gridfilepath = gridfilepath,
                info = False,
                geometry = "toroidal",
                keep_xboundaries=keep_boundaries,
                keep_yboundaries=keep_boundaries,
                )

        # Load both a case with and without guards
        if double_load is True:
            
            ds = xbout.load.open_boutdataset(
                datapath = datapath, 
                inputfilepath = inputfilepath, 
                gridfilepath = gridfilepath,
                info = False,
                geometry = "toroidal",
                keep_xboundaries=True,
                keep_yboundaries=True,
                )
            
            ds_ng = xbout.load.open_boutdataset(
                datapath = datapath, 
                inputfilepath = inputfilepath, 
                gridfilepath = gridfilepath,
                info = False,
                geometry = "toroidal",
                keep_xboundaries=False,
                keep_yboundaries=False,
                )
            
            if squeeze:
                ds = ds.squeeze(drop = True)
                ds_ng = ds_ng.squeeze(drop = True)
                
            return Case(ds, casepath), Case(ds_ng, casepath)
                
        else:
            
            # Load with guard settings as per inputs
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
            ds["Th+"].attrs.update(
                {
                "units": "eV",
                "standard_name": "ion temperature (h+)",
                "long_name": "Ion temperature (h+)",
                })

        if "Ph" in ds.data_vars:
            ds["Th"] = ds["Ph"] / (ds["Nh"] * q_e)
            ds["Th"].attrs.update(
                {
                "units": "eV",
                "standard_name": "neutra; temperature (h)",
                "long_name": "Neutral temperature (h)",
                })

        if "Pd" in ds.data_vars:
            ds["Td"] = ds["Pd"] / (ds["Nd"] * q_e)
            ds["Td"].attrs.update(
                {
                "units": "eV",
                "standard_name": "neutral temperature (d)",
                "long_name": "Neutral temperature (d)",
                })

        if "Pd+" in ds.data_vars:
            ds["Td+"] = ds["Pd+"] / (ds["Nd+"] * q_e)
            ds["Td+"].attrs.update(
               {
                "units": "eV",
                "standard_name": "ion temperature (d+)",
                "long_name": "Ion temperature (d+)",
                })

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
        
      
    
    def select_custom_core_ring(self, i):
            """
            Creates custom SOL ring slice within the core.
            i = 0 is at first domain cell.
            i = -2 is at first inner guard cell.
            i = ixseps - MXG is the separatrix.
            """
            
            if i > self.ixseps1 - self.MXG:
                raise Exception("i is too large!")
            
            selection = (slice(0+self.MXG+i,1+self.MXG+i), np.r_[slice(self.j1_2g + 1, self.j2_2g + 1), slice(self.j1_1g + 1, self.j2_1g + 1)])
            
            return self.ds.isel(x = selection[0], theta = selection[1])
        
        
        
    def select_custom_sol_ring(self, i, region):
            """
            Creates custom SOL ring slice beyond the separatrix.
            args[0] = i = index of SOL ring (0 is separatrix, 1 is first SOL ring)
            args[1] = region = all, inner, inner_lower, inner_upper, outer, outer_lower, outer_upper
            """
            
            i = i + self.ixseps1 - 1
            if i > self.nx - self.MXG*2 :
                raise Exception("i is too large!")
            
            if region == "all":
                selection = (slice(i+1,i+2), np.r_[slice(0+self.MYG, self.j2_2g + 1), slice(self.j1_1g + 1, self.nyg - self.MYG)])
            
            if region == "inner":
                selection = (slice(i+1,i+2), slice(0+self.MYG, self.ny_inner + self.MYG))
            if region == "inner_lower":
                selection = (slice(i+1,i+2), slice(0+self.MYG, int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g +2))
            if region == "inner_upper":
                selection = (slice(i+1,i+2), slice(int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g, self.ny_inner + self.MYG))
            
            if region == "outer":
                selection = (slice(i+1,i+2), slice(self.ny_inner + self.MYG*3, self.nyg - self.MYG))
            if region == "outer_lower":
                selection = (slice(i+1,i+2), slice(int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g, self.nyg - self.MYG))
            if region == "outer_upper":
                selection = (slice(i+1,i+2), slice(self.ny_inner + self.MYG*3, int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g + 2))
            
            return self.ds.isel(x = selection[0], theta = selection[1])



    def select_region(self, name):
        """
        DOUBLE NULL ONLY
        Pass this tuple to a field of any parameter spanning the grid
        to select points of the appropriate region.
        Each slice is a tuple: (x slice, y slice)
        Use it as: selected_array = array[slice] where slice = (x selection, y selection) = output from this method.
        Returns sliced xarray dataset
        NOTE: Everything is optimised for reading the case with guard cells 
        """

        slices = dict()

        slices["all"] = (slice(None,None), slice(None,None))
        slices["all_noguards"] = (slice(self.MXG,-self.MXG), np.r_[slice(self.MYG,self.ny_inner-self.MYG*2), slice(self.ny_inner+self.MYG*3, self.nyg - self.MYG)])

        slices["core"] = (slice(0,self.ixseps1), np.r_[slice(self.j1_1g + 1, self.j2_1g+1), slice(self.j1_2g + 1, self.j2_2g + 1)])
        slices["core_noguards"] = (slice(self.MXG,self.ixseps1), np.r_[slice(self.j1_1g + 1, self.j2_1g+1), slice(self.j1_2g + 1, self.j2_2g + 1)])
        slices["sol"] = (slice(self.ixseps1, None), slice(0, self.nyg))
        slices["sol_noguards"] = (slice(self.ixseps1, -self.MYG), np.r_[slice(self.MYG,self.ny_inner-self.MYG*2), slice(self.ny_inner+self.MYG*3, self.nyg - self.MYG)])

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

        selection = slices[name]
        
        return self.ds.isel(x = selection[0], theta = selection[1])
    
    

    def extract_geometry(self):
        """
        Perpare geometry variables
        """
        data = self.ds
        meta = self.ds.metadata

        # self.Rxy = meta["Rxy"]    # R coordinate array
        # self.Zxy = meta["Zxy"]    # Z coordinate array
        
        
        if meta["keep_xboundaries"] is True:
            self.MXG = meta["MXG"]
        else:
            self.MXG = 0
            
        if meta["keep_yboundaries"] is True:
            self.MYG = meta["MYG"]
        else:
            self.MYG = 0
            
        self.ixseps1 = meta["ixseps1"]
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











