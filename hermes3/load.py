#!/usr/bin/env python3

from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import traceback
import platform
from datetime import datetime as dt
from scipy import stats
from boututils.datafile import DataFile
from boutdata.collect import collect
from boutdata.data import BoutData
from boutdata.squashoutput import squashoutput
import xbout

from hermes3.utils import *
from hermes3.named_selections import *
from hermes3.plotting import *
from hermes3.fluxes import *


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
                cache = False
                )

        ds = ds.squeeze(drop = True)

        return Case(ds, casepath, unnormalise_geom = True)

    def case_2D(
                casepath,
                gridfilepath,
                verbose = False, 
                squeeze = True, 
                unnormalise_geom = True,
                unnormalise = True,
                use_squash = False):

            
        loadfilepath = os.path.join(casepath, "BOUT.dmp.*.nc")
        inputfilepath = os.path.join(casepath, "BOUT.inp")
        squashfilepath = os.path.join(casepath, "BOUT.squash.nc") # Squashoutput hardcoded to this filename

        if use_squash is True:
            squash(casepath, verbose = verbose)
            loadfilepath = squashfilepath
                

        ds = xbout.load.open_boutdataset(
                        datapath = loadfilepath, 
                        inputfilepath = inputfilepath, 
                        gridfilepath = gridfilepath,
                        info = False,
                        cache = True,
                        geometry = "toroidal",
                        keep_xboundaries=True,
                        keep_yboundaries=True,
                        )
        
        if squeeze:
            ds = ds.squeeze(drop = False)
            
        return Case(ds, casepath, 
                    unnormalise_geom,
                    unnormalise = unnormalise)
        


class Case:

    def __init__(self, ds, casepath, 
                 unnormalise_geom = False,
                 unnormalise = True):

        self.ds = ds
        self.name = os.path.split(casepath)[-1]
        self.datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
        self.inputfilepath = os.path.join(casepath, "BOUT.settings")
        self.normalised_vars = []

        if "x" in self.ds.dims:
            self.is_2d = True
        else:
            self.is_2d = False

        self.ds.metadata["colors"] = ["teal", "darkorange", "firebrick", "limegreen", "deeppink", "navy", "crimson"]

        if unnormalise is True:
            self.unnormalise(unnormalise_geom)
        else:
            print("Skipping unnormalisation")
        self.derive_vars()
        self.guard_replaced = False
        
        if self.is_2d is True:
            self.extract_2d_tokamak_geometry()
            vol = self.ds.dv.values.sum()
            print(f"CHECK: Total domain volume is {vol:.3E} [m3]")
        else:
            self.extract_1d_tokamak_geometry()
            # self.clean_guards()
            self.guard_replace()
            
        # self.ds = calculate_radial_fluxes(ds)
        # self.ds = calculate_target_fluxes(ds)

    

    def unnormalise(self, unnormalise_geom):
        
        # self.ds = self.unnormalise_xhermes(self.ds)
        
        self.calc_norms()
        

        if unnormalise_geom == False:
            list_skip = ["g11", "g_22", "dx", "dy", "J"]
            print("--> g11, g_22, dx, dy and J will not be unnormalised")
        else:
            list_skip = []

        for data_var in self.norms.keys():
            # Unnormalise variables and coordinates
            if data_var in self.ds.variables or data_var in self.ds.coords:
                if data_var not in self.normalised_vars and data_var not in list_skip:
                    self.ds[data_var] = self.ds[data_var] * self.norms[data_var]["conversion"]
                    self.normalised_vars.append(data_var)
                self.ds[data_var].attrs.update(self.norms[data_var])

    def derive_vars(self):
        ds = self.ds
        m = ds.metadata
        q_e = constants("q_e")
        
        m["Pnorm"] = m["Nnorm"] * m["Tnorm"] * q_e
        
        # From Hypnotoad trim_yboundaries() in compare_grid_files
        if ds.metadata["jyseps2_1"] != ds.metadata["jyseps1_2"]:
            ds.metadata["null_config"] = "cdn"
        else:
            ds.metadata["null_config"] = "sn"
            
        ds.metadata["species"] = [x.split("P")[1] for x in self.ds.data_vars if x.startswith("P") and len(x) < 4]
        ds.metadata["charged_species"] = [x for x in ds.metadata["species"] if "e" in x or "+" in x]
        ds.metadata["ion_species"] = [x for x in ds.metadata["species"] if "+" in x]
        ds.metadata["neutral_species"] = list(set(ds.metadata["species"]).difference(set(ds.metadata["charged_species"])))
        
        ds.metadata["recycle_pair"] = dict()
        for ion in ds.metadata["ion_species"]:
            if "recycle_as" in ds.options[ion].keys():
                ds.metadata["recycle_pair"][ion] = ds.options[ion]["recycle_as"]
            else:
                print(f"No recycling partner found for {ion}")
        
        

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

    def clean_guards(self):
        
        to_clean = ["Dd_Dpar", "Ed+_iz","Ed+_rec", "Ed_Dpar", "Edd+_cx",
                    "Fd+_iz", "Fd+_rec", "Fd_Dpar", "Fdd+_cx", "Rd+_ex",
                    "Rd+_rec", "Sd+_iz", "Sd+_rec", "Sd+_src", "Sd_Dpar",
                    "Sd_src"]
        
        for param in to_clean:
            self.ds[param]
        
        
    def guard_replace(self):
        pass
        if self.is_2d == False:
            if self.ds.metadata["keep_yboundaries"] == 1:
                # Replace inner guard cells with values at cell boundaries
                # Hardcoded dimension order: t, y
                # Cell order at target:
                # ... | last | guard | second guard
                #            ^target   ^not used
                #     |  -3  |  -2   |      -1

                if self.guard_replaced == False:
                    for var_name in self.ds.data_vars:
                        var = self.ds[var_name]
                        
                        if "pos" in var.dims:
                            var[{"pos":-2}] = (var[{"pos":-3}] + var[{"pos":-2}])/2
                            var[{"pos":1}] = (var[{"pos":1}] + var[{"pos":2}])/2
                            
                else:
                    raise Exception("Guards already replaced!")
                        
                self.guard_replaced = True
            else:
                raise Exception("Y guards are missing from file!")
        else:
            raise Exception("2D guard replacement not done yet!")


    def unnormalise_xhermes(self, ds):
        # Normalisation
        meta = ds.attrs["metadata"]
        Cs0 = meta["Cs0"]
        Omega_ci = meta["Omega_ci"]
        rho_s0 = meta["rho_s0"]
        Nnorm = meta["Nnorm"]
        Tnorm = meta["Tnorm"]

        # SI values
        Mp = 1.67e-27  # Proton mass
        e = 1.602e-19  # Coulombs

        # Coordinates
        ds.t.attrs["units_type"] = "hermes"
        ds.t.attrs["units"] = "s"
        ds.t.attrs["conversion"] = 1.0 / Omega_ci

        for varname in ds:
            da = ds[varname]
            if len(da.dims) == 4:  # Time-evolving field
                da.attrs["units_type"] = "hermes"

                # Check if data already has units and conversion attributes
                if ("units" in da.attrs) and ("conversion" in da.attrs):
                    print(varname + " already annotated")
                    continue  # No need to add attributes

                # Mark as Hermes-normalised data
                da.attrs["units_type"] = "hermes"

                if varname[:2] == "NV":
                    # Momentum
                    da.attrs.update(
                        {
                            "units": "kg m / s",
                            "conversion": Mp * Nnorm * Cs0,
                            "standard_name": "momentum",
                            "long_name": varname[2:] + " parallel momentum",
                        }
                    )
                elif varname[0] == "N":
                    # Density
                    da.attrs.update(
                        {
                            "units": "m^-3",
                            "conversion": Nnorm,
                            "standard_name": "density",
                            "long_name": varname[1:] + " number density",
                        }
                    )

                elif varname[0] == "T":
                    # Temperature
                    da.attrs.update(
                        {
                            "units": "eV",
                            "conversion": Tnorm,
                            "standard_name": "temperature",
                            "long_name": varname[1:] + " temperature",
                        }
                    )

                elif varname[0] == "V":
                    # Velocity
                    da.attrs.update(
                        {
                            "units": "m / s",
                            "conversion": Cs0,
                            "standard_name": "velocity",
                            "long_name": varname[1:] + " parallel velocity",
                        }
                    )

                elif varname[0] == "P":
                    # Pressure
                    da.attrs.update(
                        {
                            "units": "Pa",
                            "conversion": e * Tnorm * Nnorm,
                            "standard_name": "pressure",
                            "long_name": varname[1:] + " pressure",
                        }
                    )
                elif varname == "phi":
                    # Potential
                    da.attrs.update(
                        {
                            "units": "V",
                            "conversion": Tnorm,
                            "standard_name": "potential",
                            "long_name": "Plasma potential",
                        }
                    )
                else:
                    # Don't know what this is
                    da.attrs["units_type"] = "unknown"

        if ds.attrs["options"] is not None:
            # Process options
            options = ds.attrs["options"]

            # List of components
            component_list = [
                c.strip() for c in options["hermes"]["components"].strip(" ()\t").split(",")
            ]

            # Turn into a dictionary
            components = {}
            for component in component_list:
                if component in options:
                    c_opts = options[component]
                    if "type" in c_opts:
                        c_type = c_opts["type"]
                    else:
                        c_type = component  # Type is the name of the component
                else:
                    c_opts = None
                    c_type = component
                components[component] = {"type": c_type, "options": c_opts}
            ds.attrs["components"] = components

        return ds
    
    
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
        
        "g22": {
            "conversion": 1/(m["rho_s0"] * m["rho_s0"]),
            "units": "m-2",
            "standard_name": "g22",
            "long_name": "g22, 1/h_theta^2",
        },
        
        "g_22": {
            "conversion": m["rho_s0"] * m["rho_s0"],
            "units": "m2",
            "standard_name": "g_22",
            "long_name": "g_22, B^2*h_theta^2/Bpol^2",
        },
        
        "g_33": {
            "conversion": m["rho_s0"] * m["rho_s0"],
            "units": "m2",
            "standard_name": "g_22",
            "long_name": "g_22, R^2",
        },
        
        "g11": {
            "conversion": (m["Bnorm"] * m["rho_s0"])**2,
            "units": "T-2m-2",
            "standard_name": "g11",
            "long_name": "g11",
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
        
        "Td": {
            "conversion": m["Tnorm"],
            "units": "eV",
            "standard_name": "neutral temperature (d)",
            "long_name": "Neutral temperature (d)",
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
        
        "Pd": {
            "conversion": m["Nnorm"] * m["Tnorm"] * q_e,
            "units": "Pa",
            "standard_name": "neutral pressure (d)",
            "long_name": "Neutral pressure (d)"
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
        
        "Pd_src": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "neutral energy source (d)",
            "long_name": "Neutral energy source (d)"
        },
        
        "Pe_src": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "electron energy source (d)",
            "long_name": "Electron energy source (d)"
        },
        
        "SPd+": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "d+ net pressure source",
            "long_name": "d+ net pressure source"
        },
        
        "SPe": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "e net pressure source",
            "long_name": "e net pressure source"
        },
        
        "Sd_src": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "neutral density source (d)",
            "long_name": "Neutral density source (d)"
        },
        
        "Sd+_src": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "neutral density source (d)",
            "long_name": "Neutral density source (d)"
        },

        "Rd+_ex": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "exciation radiation (d+)",
            "long_name": "Multi-step ionisation radiation (d+)"
        },
        
        "Rd+_rec": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "recombination radiation (d+)",
            "long_name": "Recombination radiation (d+)"
        },
        
        "Ed+_rec": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "recombination plasma energy source (d+)",
            "long_name": "Recombination plasma energy source (d+)"
        },
        
        "Ed+_iz": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "ionisation plasma energy source (d+)",
            "long_name": "Ionization plasma energy source (d+)"
        },
        
        "Edd+_cx": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "ionisation plasma energy source (d+)",
            "long_name": "Ionization plasma energy source (d+)"
        },
        
        "Rar": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "argon radiation",
            "long_name": "Argon radiation"
        },
        
        "Rc": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "argon radiation",
            "long_name": "Argon radiation"
        },
        
        "Rne": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "argon radiation",
            "long_name": "Argon radiation"
        },
        
        "Rn": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "argon radiation",
            "long_name": "Argon radiation"
        },


        "Sd+_iz": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units": "m-3s-1",
            "standard_name": "ionisation",
            "long_name": "Ionisation ion source (d+)"
        },
        
        "Sd+_rec": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units": "m-3s-1",
            "standard_name": "recombination",
            "long_name": "Recombination ion source (d+)"
        },
        
        "Sd+_feedback": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units": "m-3s-1",
            "standard_name": "density source",
            "long_name": "Upstream density feedback source"
        },
        
        "density_source_shape_d+": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units": "m-3s-1",
            "standard_name": "density source shape",
            "long_name": "Upstream density feedback source shape"
        },
        
        "Sd_pfr_recycle": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units":"m-3s-1",
            "standard_name": "PFR recycle neutral density source (d)",
            "long_name": "PFR recycling neutral density source (d)"
        },
        
        "Sd_sol_recycle": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units":"m-3s-1",
            "standard_name": "SOL recycle neutral density source (d)",
            "long_name": "SOL recycling neutral density source (d)"
        },
        
        "Sd_wall_recycle": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units":"m-3s-1",
            "standard_name": "wall recycle neutral density source (d)",
            "long_name": "wall recycling neutral density source (d)"
        },
        
        "Sd_pump": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units":"m-3s-1",
            "standard_name": "pump recycle neutral density source (d)",
            "long_name": "pump recycling neutral density source (d)"
        },
        
        "Sd_target_recycle": {
            "conversion": m["Nnorm"] * m["Omega_ci"],
            "units":"m-3s-1",
            "standard_name": "Target recycle neutral density source (d)",
            "long_name": "Target recycling neutral density source (d)"
        },
        
        "Ed_pfr_recycle": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "PFR recycle neutral energy source",
            "long_name": "PFR recycle neutral energy source"
        },
        
        "Ed_sol_recycle": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "SOL recycle neutral energy source",
            "long_name": "SOL recycle neutral energy source"
        },
        
        "Ed_target_recycle": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "Target recycle neutral energy source",
            "long_name": "Target recycle neutral energy source"
        },
        
        "Ed_wall_refl": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "Wall reflection neutral energy source",
            "long_name": "Wall reflection neutral energy source"
        },
        
        "Ed_target_refl": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "Target reflection neutral energy source",
            "long_name": "Target reflection neutral energy source"
        },
        
        "Ed_pump": {
            "conversion": q_e * m["Nnorm"] * m["Tnorm"] * m["Omega_ci"],
            "units": "Wm-3",
            "standard_name": "Pump neutral energy source",
            "long_name": "Pump neutral energy source"
        },
        
        "ParticleFlow_d+_xlow": {
            "conversion": m["rho_s0"] * m["rho_s0"]**2 * m["Nnorm"] * m["Omega_ci"],
            "units":"s-1",
            "standard_name": "X flow of d+",
            "long_name": "X flow of d+"
        },
        
        "ParticleFlow_d+_ylow": {
            "conversion": m["rho_s0"] * m["rho_s0"]**2 * m["Nnorm"] * m["Omega_ci"],
            "units":"s-1",
            "standard_name": "Y flow of d+",
            "long_name": "Y flow of d+"
        },
        
        "EnergyFlow_d+_xlow": {
            "conversion": m["rho_s0"] * m["rho_s0"]**2 * m["Nnorm"] * m["Tnorm"] * constants("q_e") * m["Omega_ci"],
            "units":"W",
            "standard_name": "X flow of d+ energy",
            "long_name": "X flow of d+ energy"
        },
        
        "EnergyFlow_e_xlow": {
            "conversion": m["rho_s0"] * m["rho_s0"]**2 * m["Nnorm"] * m["Tnorm"] * constants("q_e") * m["Omega_ci"],
            "units":"W",
            "standard_name": "X flow of e energy",
            "long_name": "X flow of e energy"
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

        "Fdd+_cx": {
            "conversion": constants("mass_p") * m["Nnorm"] * m["Cs0"] * m["Omega_ci"],
            "units": "kgm-3s-2",
            "standard_name": "CX momentum transfer",
            "long_name": "CX momentum transfer"
        },
        
        "Fd+_iz": {
            "conversion": constants("mass_p") * m["Nnorm"] * m["Cs0"] * m["Omega_ci"],
            "units": "kgm-3s-2",
            "standard_name": "IZ momentum transfer",
            "long_name": "IZ momentum transfer"
        },
        
        "Fd+_rec": {
            "conversion": constants("mass_p") * m["Nnorm"] * m["Cs0"] * m["Omega_ci"],
            "units": "kgm-3s-2",
            "standard_name": "Rec momentum transfer",
            "long_name": "Rec momentum transfer"
        },
        
        "SNVd+": {
            "conversion": constants("mass_p") * m["Nnorm"] * m["Cs0"] * m["Omega_ci"],
            "units": "kgm-3s-2",
            "standard_name": "Net momentum transfer",
            "long_name": "Net momentum transfer"
        },
        
        "Vd": {
            "conversion": m["Cs0"],
            "units": "ms-1",
            "standard_name": "neutral velocity",
            "long_name": "Neutral velocity (d+)"
        },
        
        "Vd+": {
            "conversion": m["Cs0"],
            "units": "ms-1",
            "standard_name": "ion velocity",
            "long_name": "Ion velocity (d+)"
        },
        
        "anomalous_D_e": {
            "conversion": m["rho_s0"] * m["rho_s0"] * m["Omega_ci"],
            "units": "m2s-1",
            "standard_name": "anomalous density diffusion (e)",
            "long_name": "anomalous density diffusion (e)"
        },
        
        "anomalous_D_d+": {
            "conversion": m["rho_s0"] * m["rho_s0"] * m["Omega_ci"],
            "units": "m2s-1",
            "standard_name": "anomalous density diffusion (d+)",
            "long_name": "anomalous density diffusion (d+)"
        },
        
        "anomalous_Chi_e": {
            "conversion": m["rho_s0"] * m["rho_s0"] * m["Omega_ci"],
            "units": "m2s-1",
            "standard_name": "anomalous thermal diffusion (e)",
            "long_name": "anomalous thermal diffusion (e)"
        },
        
        "anomalous_Chi_d+": {
            "conversion": m["rho_s0"] * m["rho_s0"] * m["Omega_ci"],
            "units": "m2s-1",
            "standard_name": "anomalous thermal diffusion (d+)",
            "long_name": "anomalous thermal diffusion (d+)"
        },
        
        "Dnnd": {
            "conversion": m["rho_s0"] * m["rho_s0"] * m["Omega_ci"],
            "units": "m2s-1",
            "standard_name": "Neutral diffusion (d)",
            "long_name": "Neutral diffusion (d)"
        },
        
        "t": {
            "conversion": 1/m["Omega_ci"],
            "units": "s",
            "standard_name": "time",
            "long_name": "Time"
        },
        

        
        
        }

        self.norms = d
        
    def select_symmetric_puff(self, width, center_half_gap):
        """
        Select region meant for setting outboard neutral puff.
        The region is a poloidal row of cells in the radial coordinate
        of the final radial fluid cell.
        There are two puffs symmetric about the midplane axis.
        
        Parameters:
            - width: size of each puff region in no. of cells
            - center_half_gap: half of the gap between the puffs in no. of cells
        """
        
        # width = 3
        # center_half_gap = 1

        midplane_a = int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g
        midplane_b = int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g + 1

        selection =  (-self.MXG-1, 
                    np.r_[
                        slice(midplane_b+center_half_gap, midplane_b+center_half_gap+width),
                        slice(midplane_b-center_half_gap-width, midplane_b-center_half_gap),
                        ])

        return self.ds.isel(x = selection[0], theta = selection[1])
    
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
            
            NOTE: INDEX HERE IS THE ACTUAL INDEX AS OPPOSED TO THE CUSTOM CORE RING
            
            TODO: CHECK THE OFFSETS ON X AXIS, THEY ARE POTENTIALLY WRONG
            """
            
            # if i < self.ixseps1 - self.MXG*2 :
            #     raise Exception("i is too small!")
            if i > self.nx - self.MXG*2 :
                raise Exception("i is too large!")
            
            if self.ds.metadata["topology"] == "connected-double-null":
                
                outer_midplane_a = int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g
                outer_midplane_b = int((self.j2_2g - self.j1_2g) / 2) + self.j1_2g + 1     
                inner_midplane_a = int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g 
                inner_midplane_b = int((self.j2_1g - self.j1_1g) / 2) + self.j1_1g + 1               
                
                
                
                if region == "all":
                    selection = (slice(i+1,i+2), np.r_[slice(0+self.MYG, self.j2_2g + 1), slice(self.j1_1g + 1, self.nyg - self.MYG)])
                
                if region == "inner":
                    selection = (slice(i+1,i+2), slice(0+self.MYG, self.ny_inner + self.MYG))
                if region == "inner_lower":
                    selection = (slice(i+1,i+2), slice(0+self.MYG, inner_midplane_a +1))
                if region == "inner_upper":
                    selection = (slice(i+1,i+2), slice(inner_midplane_b, self.ny_inner + self.MYG))
                
                if region == "outer":
                    selection = (slice(i+1,i+2), slice(self.ny_inner + self.MYG*3, self.nyg - self.MYG))
                if region == "outer_lower":
                    selection = (slice(i+1,i+2), slice(outer_midplane_b, self.nyg - self.MYG))
                if region == "outer_upper":
                    selection = (slice(i+1,i+2), slice(self.ny_inner + self.MYG*3, outer_midplane_a+1))
                    
            elif self.ds.metadata["topology"] == "single-null":
                
                if region == "all":
                    selection = (slice(i+0,i+1), slice(0+self.MYG, self.nyg - self.MYG))
            
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
        
        if self.MXG != 0:
            
            slices["outer_sol_edge"] = (slice(-1 - self.MXG, - self.MXG), slice(self.ny_inner+self.MYG*3, self.nyg - self.MYG))
            slices["inner_sol_edge"] = (slice(-1 - self.MXG, - self.MXG), slice(self.MYG, self.ny_inner+self.MYG))
            slices["sol_edge"] = (slice(-1 - self.MXG, - self.MXG), np.r_[slice(self.j1_1g + 1, self.j2_1g + 1), slice(self.ny_inner+self.MYG*3, self.nyg - self.MYG)])
            
        else:
            
            slices["outer_sol_edge"] = (slice(-1, None), slice(self.ny_inner+self.MYG*3, self.nyg - self.MYG))
            slices["inner_sol_edge"] = (slice(-1, None), slice(self.MYG, self.ny_inner+self.MYG))
            slices["sol_edge"] = (slice(-1 - self.MXG, - self.MXG), np.r_[slice(self.j1_1g + 1, self.j2_1g + 1), slice(self.ny_inner+self.MYG*3, self.nyg - self.MYG)])
        
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
    
    def select_domain_boundary(self):
        """
        Extract R,Z coordinates of the model boundary using cell corner information
        """
        r = []
        z = []

        loc = {
            "inner_lower_target" : "xy_lower_left_corners",
            "inner_sol_edge" : "xy_lower_right_corners",
            "inner_upper_target" : "xy_upper_right_corners",
            "upper_pfr_edge" : "xy_upper_left_corners",
            "outer_upper_target" : "xy_lower_left_corners",
            "outer_sol_edge" : "xy_lower_right_corners",
            "outer_lower_target" : "xy_upper_right_corners",
            "lower_pfr_edge" : "xy_upper_left_corners"
        }

        for region in loc.keys():
            sel = self.select_region(region)

            name = loc[region]
            r.append(sel[f"R{name}"].values.flatten())
            z.append(sel[f"Z{name}"].values.flatten())

        r = np.concatenate(r)
        z = np.concatenate(z)
        
        return r,z
        
    
    def extract_1d_tokamak_geometry(self):
        ds = self.ds
        meta = self.ds.metadata

        # Reconstruct grid position from dy
        dy = ds.coords["dy"].values
        n = len(dy)
        pos = np.zeros(n)
        pos[0] = -0.5*dy[1]
        pos[1] = 0.5*dy[1]

        for i in range(2,n):
            pos[i] = pos[i-1] + 0.5*dy[i-1] + 0.5*dy[i]
            
        # pos = ds.coords["y"].values.copy()

        # Guard replace to get position at boundaries
        pos[-2] = (pos[-3] + pos[-2])/2
        pos[1] = (pos[1] + pos[2])/2 

        # Set 0 to be at first cell boundary in domain
        pos = pos - pos[1]
        self.pos = pos
        self.ds["pos"] = (["y"], pos)
        self.ds = self.ds.swap_dims({"y":"pos"})
        self.ds.coords["pos"].attrs = self.ds.coords["y"].attrs

        # Replace y in dataset with the new one
        # ds.coords["y"] = pos
        
        self.ds["da"] = self.ds.J / np.sqrt(ds.g_22)
        
        self.ds["da"].attrs.update({
            "conversion" : 1,
            "units" : "m2",
            "standard_name" : "cross-sectional area",
            "long_name" : "Cell parallel cross-sectional area"})
        
        self.ds["dv"] = self.ds.J * self.ds.dy
        self.ds["dv"].attrs.update({
            "conversion" : 1,
            "units" : "m3",
            "standard_name" : "cell volume",
            "long_name" : "Cell Volume"})


    def extract_2d_tokamak_geometry(self):
        """
        Perpare geometry variables
        """
        ds = self.ds
        m = self.ds.metadata
        
        # Add theta index to coords so that both X and theta can be accessed index-wise
        # It is surprisingly hard to extract the index of coordinates in Xarray...
        ds.coords["theta_idx"] = (["theta"], range(len(ds.coords["theta"])))
        
        if "single-null" in ds.metadata["topology"]:
            ds.metadata["targets"] = ["inner_lower", "outer_lower"]
        elif "double-null" in ds.metadata["topology"]:
            ds.metadata["targets"] = ["inner_lower", "outer_lower", "inner_upper", "outer_upper"]
            
        num_targets = len(ds.metadata["targets"])

        # self.Rxy = meta["Rxy"]    # R coordinate array
        # self.Zxy = meta["Zxy"]    # Z coordinate array
        
        
        if m["keep_xboundaries"] == 0:
            m["nxg"] = m["nx"] - m["MXG"] * 2  # Take guards into account
            m["MXG"] = 0
        else:
            m["nxg"] = m["nx"]
            
        if m["keep_yboundaries"] == 0:
            m["nyg"] = m["ny"]    # Already doesn't include guard cells
            m["MYG"] = 0
        else:
            m["nyg"] = m["ny"] + m["MYG"] * num_targets   # ny taking guards into account
            
            
          
        # nyg, nxg: cell counts which are always with guard cells if they exist, or not if they don't
              
        
        m["nyg"] = m["ny"] + m["MYG"] * num_targets   # ny taking guards into account
            
        m["j1_1"] = m["jyseps1_1"]
        m["j1_2"] = m["jyseps1_2"]
        m["j2_1"] = m["jyseps2_1"]
        m["j2_2"] = m["jyseps2_2"]
        
        m["j1_1g"] = m["j1_1"] + m["MYG"]
        m["j1_2g"] = m["j1_2"] + m["MYG"] * (num_targets - 1)
        m["j2_1g"] = m["j2_1"] + m["MYG"]
        m["j2_2g"] = m["j2_2"] + m["MYG"] * (num_targets - 1)
        
        
        
        
        # print(m["nxg"])
        # print(m["nx"])
            
        # Array of radial (x) indices and of poloidal (y) indices for each cell
        ds["x_idx"] = (["x", "theta"], np.array([np.array(range(m["nxg"]))] * int(m["nyg"])).transpose())
        ds["y_idx"] = (["x", "theta"], np.array([np.array(range(m["nyg"]))] * int(m["nxg"])))
        
        # Cell areas in flux space
        ds["dv"] = (["x", "theta"], ds["dx"].data * ds["dy"].data * ds["dz"].data * ds["J"].data)
        ds["dv"].attrs.update({
            "conversion" : 1,
            "units" : "m3",
            "standard_name" : "cell volume",
            "long_name" : "Cell volume",
            "source" : "xHermes"})
        
        # Cell areas in real space - comes from Jacobian
        # TODO: Check these against dx/dy to ensure volume is the same
        # dV = (hthe/Bpol) * (R*Bpol*dr) * dy*2pi = hthe * dy * dr * 2pi * R
        # self.dr = self.dx / (self.ds.R * self.ds.Bpxy)    # Length of cell in radial direction
        # self.hthe = self.J * self.ds["Bpxy"]    # poloidal arc length per radian
        # self.dl = self.dy * self.hthe    # poloidal arc length
        
        ds["dr"] = (["x", "theta"], ds.dx.data / (ds.R.data * ds.Bpxy.data))  # eqv. to sqrt(g11)
        ds["dr"].attrs.update({
            "conversion" : 1,
            "units" : "m",
            "standard_name" : "radial length",
            "long_name" : "Length of cell in the radial direction",
            "source" : "xHermes"})
        
        ds["hthe"] = (["x", "theta"], ds["J"].data * ds["Bpxy"].data)    # h_theta
        ds["hthe"].attrs.update({
            "conversion" : 1,
            "units" : "m/radian",
            "standard_name" : "h_theta: poloidal arc length per radian",
            "long_name" : "h_theta: poloidal arc length per radian",
            "source" : "xHermes"})
        
        ds["dl"] = (["x", "theta"], ds["dy"].data * ds["hthe"].data)    # poloidal arc length
        ds["dl"].attrs.update({
            "conversion" : 1,
            "units" : "m",
            "standard_name" : "poloidal arc length",
            "long_name" : "Poloidal arc length",
            "source" : "xHermes"})
        
        ds["dtor"] = (["x", "theta"], ds["dz"].data * np.sqrt(ds["g_33"].data))   # Toroidal length
        ds["dtor"].attrs.update({
            "conversion" : 1,
            "units" : "m",
            "standard_name" : "Toroidal length",
            "long_name" : "Toroidal length",
            "source" : "xHermes"})


    # def extract_2d_tokamak_geometry(self):
    #     """
    #     Perpare geometry variables
    #     """
    #     data = self.ds
    #     meta = self.ds.metadata

    #     # self.Rxy = meta["Rxy"]    # R coordinate array
    #     # self.Zxy = meta["Zxy"]    # Z coordinate array
        
        
    #     if meta["keep_xboundaries"] == 1:
    #         self.MXG = meta["MXG"]
    #     else:
    #         self.MXG = 0
            
    #     if meta["keep_yboundaries"] == 1:
    #         self.MYG = meta["MYG"]
    #     else:
    #         self.MYG = 0
            
    #     self.ixseps1 = meta["ixseps1"]
    #     self.ny_inner = meta["ny_inner"]
    #     self.ny = meta["ny"]
    #     self.nyg = self.ny + self.MYG * 4 # with guard cells
    #     self.nx = meta["nx"]
        
    #     # Array of radial (x) indices and of poloidal (y) indices in the style of Rxy, Zxy
    #     self.x_idx = np.array([np.array(range(self.nx))] * int(self.nyg)).transpose()
    #     self.y_idx = np.array([np.array(range(self.nyg))] * int(self.nx))
        
    #     self.yflat = self.y_idx.flatten()
    #     self.xflat = self.x_idx.flatten()
    #     self.rflat = self.ds.coords["R"].values.flatten()
    #     self.zflat = self.ds.coords["Z"].values.flatten()

    #     self.j1_1 = meta["jyseps1_1"]
    #     self.j1_2 = meta["jyseps1_2"]
    #     self.j2_1 = meta["jyseps2_1"]
    #     self.j2_2 = meta["jyseps2_2"]
    #     self.ixseps2 = meta["ixseps2"]
    #     self.ixseps1 = meta["ixseps1"]
    #     self.Rxy = self.ds.coords["R"]
    #     self.Zxy = self.ds.coords["Z"]

    #     self.j1_1g = self.j1_1 + self.MYG
    #     self.j1_2g = self.j1_2 + self.MYG * 3
    #     self.j2_1g = self.j2_1 + self.MYG
    #     self.j2_2g = self.j2_2 + self.MYG * 3
        
    #     # Cell areas in flux space
    #     # dV = dx * dy * dz * J where dz is assumed to be 2pi in 2D
    #     self.dx = data["dx"]
    #     self.dy = data["dy"]
    #     self.dydx = data["dy"] * data["dx"]    # Poloidal surface area
    #     self.J = data["J"]
    #     dz = 2*np.pi    # Axisymmetric
    #     self.dv = self.dydx * dz * data["J"]    # Cell volume
        
    #     # Cell areas in real space
    #     # TODO: Check these against dx/dy to ensure volume is the same
    #     # dV = (hthe/Bpol) * (R*Bpol*dr) * dy*2pi = hthe * dy * dr * 2pi * R
    #     self.dr = self.dx / (self.ds.R * self.ds.Bpxy)    # Length of cell in radial direction
    #     self.hthe = self.J * self.ds["Bpxy"]    # poloidal arc length per radian
    #     self.dl = self.dy * self.hthe    # poloidal arc length
        
    #     self.ds["dr"] = self.ds.dx / (self.ds.R * self.ds.Bpxy)
    #     self.ds["dr"].attrs.update({
    #         "conversion" : 1,
    #         "units" : "m",
    #         "standard_name" : "radial length",
    #         "long_name" : "Length of cell in the radial direction"})
        
    #     self.ds["hthe"] = self.ds.J * self.ds["Bpxy"]    # h_theta
    #     self.ds["hthe"].attrs.update({
    #         "conversion" : 1,
    #         "units" : "m/radian",
    #         "standard_name" : "h_theta: poloidal arc length per radian",
    #         "long_name" : "h_theta: poloidal arc length per radian"})
        
    #     self.ds["dl"] = self.ds.dy * self.ds["hthe"]    # poloidal arc length
    #     self.ds["dl"].attrs.update({
    #         "conversion" : 1,
    #         "units" : "m",
    #         "standard_name" : "poloidal arc length",
    #         "long_name" : "Poloidal arc length"})
        
    #     self.ds["dv"] = self.dydx * dz * data["J"]    # Cell volume
    #     self.ds["dv"].attrs.update({
    #         "conversion" : 1,
    #         "units" : "m3",
    #         "standard_name" : "cell volume",
    #         "long_name" : "Cell volume"})
        
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





def squash(casepath, verbose = True, force = False):
    """
    Checks if squashed file exists. If it doesn't, or if it's older than the dmp files, 
    then it creates a new squash file.
    
    Inputs
    ------
    Casepath is the path to the case directory
    verbose gives you extra info prints
    force always recreates squash file
    """
    
    datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
    inputfilepath = os.path.join(casepath, "BOUT.inp")
    squashfilepath = os.path.join(casepath, "BOUT.squash.nc") # Squashoutput hardcoded to this filename

    recreate = True if force is True else False   # Override to recreate squashoutput
    squash_exists = False
    
    if verbose is True: print(f"- Looking for squash file")
        
    if "BOUT.squash.nc" in os.listdir(casepath):  # Squash file found?
        
        squash_exists = True
        
        squash_date = os.path.getmtime(squashfilepath)
        dmp_date = os.path.getmtime(os.path.join(casepath, "BOUT.dmp.0.nc"))
        
        squash_date_string = dt.strftime(dt.fromtimestamp(squash_date), r"%m/%d/%Y, %H:%M:%S")
        dmp_date_string = dt.strftime(dt.fromtimestamp(dmp_date), r"%m/%d/%Y, %H:%M:%S")
        
        if verbose is True: print(f"- Squash file found. squash date {squash_date_string}, dmp file date {dmp_date_string}") 
        
        if dmp_date > squash_date:   #Recreate if squashoutput is too old
            recreate = True
            print(f"- dmp files are newer than the squash file! Recreating...") 
            
    else:
        if verbose is True: print(f"- Squashoutput file not found, creating...")
        recreate = True
        

    if recreate is True:
        
        if squash_exists is True:  # Squashoutput will not overwrite, so we delete the file first
            os.remove(squashfilepath)
            
        squashoutput(
            datadir = casepath,
            outputname = squashfilepath,
            xguards = True,   # Take all xguards
            yguards = "include_upper",  # Take all yguards (yes, confusing name)
            parallel = False,   # Seems broken atm
            quiet = verbose
        )
        
        if verbose is True: print(f"- Done")
            





