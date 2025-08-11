from boututils.datafile import DataFile
from boutdata.collect import collect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pathlib
import platform
import traceback
import xarray as xr
import xbout
import scipy
import re
import netCDF4 as nc
import pickle
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

onedrive_path = onedrive_path = str(os.getcwd()).split("OneDrive")[0] + "OneDrive"
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\gridtools"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\sdtools"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\soledge"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages"))


# from gridtools.hypnotoad_tools import *
# from gridtools.b2_tools import *
# from gridtools.utils import *

from hermes3.case_db import *
from hermes3.load import *
from hermes3.named_selections import *
from hermes3.plotting import *
from hermes3.grid_fields import *
from hermes3.accessors import *
from hermes3.utils import *
# from code_comparison.viewer_2d import *
# from code_comparison.code_comparison import *

# import gridtools.solps_python_scripts.setup
# from gridtools.solps_python_scripts.plot_solps       import plot_1d, plot_2d, plot_wall_loads
# from gridtools.solps_python_scripts.read_ft44 import read_ft44
import ipywidgets as widgets

# from solps_python_scripts.read_b2fgmtry import read_b2fgmtry



class SOLPScase():
    def __init__(self, path):
        
        """
        Note that everything in the balance file is in the Y, X convention unlike the
        X, Y convention in the results. This is not intended and it reads as X, Y in MATLAB.
        You MUST transpose everything so that it's Y,X (large then small number) so it's consistent with geometry
        which is the opposite. This should already be done in this script.
        
        NOTES ON GUARD CELLS IN SOLPS
        - Total cell count is nx+2, ny+2. Guard cells are there for upper targets but somehow don't get counted (?!). 
          All of the balance file results are with guard cells and this function plots them by default.
          Their position/mask can be found in resignore available in b2fgmtry. 
        - Guard cells are always really tiny!!!
        
        Guide on radiation from DM
        b2stel_she_bal in balance.nc gives you the radiation from each ionisation state. 
        You can get the species indeces from the string matrix "species" 
        (for Ryoko's case I think it's 4:21 for the neon ions but do check)
        numbers are in W, so divide by "vol" to get W/m3
        
        Balance file variables
        
        FLUXES
        ------------
        ALL ARE IN W OR S^-1 and defined at LHS cell edge
        Many variable names are constructed from locations/concepts defined in the manual (e.g. emel)
        ALL fluxes have dimensions: y (poloidal), x (radial), direction (0=pol,1=rad), species
        
        Particle fluxes:
        fna is particle flux. fna_pll is the parallel advection. 
        
        Heat fluxes:
        The third dimension is the direction, 0 = poloidal, 1 = radial
        fhi_32 is the convective ion heat flux (52 is only if you use total energy)
        fhi_cond is conductive
        fhe_thermj is the parallel current electron heat flux
        Rest are to do with drifts or currents
        
        fht = total heat flux = fhe_cond + 5/3*fhe_32 + fhi_cond + 5/3*fhi_32 + fhe_thermj
        
        BALANCES
        ----------
        b2stel_she_bal: radiation losses due to atomic processes in W/m3. Third dimension is species index.
        eirene_mc_emel_she_bal: Electron energy loss/gain due to interaction with molecules. 3rd dim is strata
        eirene_mc_empl_shi_bal: Ion energy loss/gain due to interaction with molecules. 3rd dim is strata
        
        eirene_mc_papl_sna_bal: particle source due to ionisation in s^-1. 3rd dim species (second is EIRENE neutral), 4d dim strata
        
        """
        self.path = path
        raw_balance = nc.Dataset(os.path.join(path, "balance.nc"))
        
        ## Need to transpose to get over the backwards convention compared to MATLAB
        bal = {}
        for key in raw_balance.variables:
            bal[key] = raw_balance[key][:].transpose()
            
        self.bal = bal
        
        raw_balance.close()

        self.params = list(bal.keys())
        self.params.sort()
        self.params_original = self.params.copy()
        
        # Set up geometry
        
        g = self.g = {}
        
        # Get cell centre coordinates
        R = self.bal["R"] = self.g["R"] = np.mean(bal["crx"], axis=2)
        Z = self.bal["Z"] = self.g["Z"] = np.mean(bal["cry"], axis=2)
        self.g["hx"] = bal["hx"]
        
        if "hy" in bal:
            self.g["hy"] = bal["hy"]
        else:
            print("Warning, hy not found in balance file")
        self.g["crx"] = bal["crx"]
        self.g["cry"] = bal["cry"]
        self.g["nx"] = self.g["crx"].shape[0]
        self.g["ny"] = self.g["crx"].shape[1]
        self.g["vol"] = self.bal["vol"]
        
        self.bal["Btot"] = self.g["Btot"] = bal["bb"][:,:,3]
        self.bal["Bpol"] = self.g["Bpol"] = bal["bb"][:,:,0]
        self.bal["Btor"] = self.g["Btor"] = np.sqrt(self.g["Btot"]**2 - self.g["Bpol"]**2) 
        
        
        ## Get branch cuts
        leftix = bal["leftix"]+1
        leftix_diff = np.diff(leftix[:,1])
        g["xcut"] = xcut = np.argwhere(leftix_diff<0).squeeze()
        g["leftcut"] = [xcut[0]-2, xcut[1]+2]
        g["rightcut"] = [xcut[4]-1, xcut[3]-1]
        
        omp = self.g["omp"] = int((g["rightcut"][0] + g["rightcut"][1])/2) + 1
        imp = self.g["imp"] = int((g["leftcut"][0] + g["leftcut"][1])/2)
        upper_break = self.g["upper_break"] = g["xcut"][2] + 1
        sep = self.g["sep"] = bal["jsep"][0] + 2  # Add 1 for guard cell and another to get first ring AFTER separatrix
        # self.g["xpoints"] = self.find_xpoints()   # NOTE: Not robust
        
        
        ## Prepare selectors
        ## ALL SELECTORS HAVE GUARD CELLS   
        psel = {}
        psel["inner_lower_target"] = 1
        psel["inner_upper_target"] = upper_break -2
        psel["outer_upper_target"] = upper_break+1
        psel["outer_lower_target"] = -2

        psel["inner_lower_target_guard"] = 0
        psel["inner_upper_target_guard"] = upper_break -1
        psel["outer_upper_target_guard"] = upper_break
        psel["outer_lower_target_guard"] = -1

        psel["outer_upper_leg"] = slice(self.g["upper_break"]+1, self.g["rightcut"][1]+2)
        psel["inner_upper_leg"] = slice(self.g["leftcut"][1]-2, self.g["upper_break"]-1)
        psel["outer_lower_leg"] = slice(self.g["rightcut"][0]+2, -1)
        psel["inner_lower_leg"] = slice(0, self.g["leftcut"][0]+2)
        psel["omp"] = omp
        psel["imp"] = imp
        
        ## X-point indices: first cell downstream of X-point
        psel["inner_lower_xpoint"] = g["xcut"][0] -1
        psel["inner_upper_xpoint"] = g["xcut"][1]
        psel["outer_upper_xpoint"] = g["xcut"][3]
        psel["outer_lower_xpoint"] = g["xcut"][4]+1
        
        # These are for 1d poloidal data getter
        psel["inner_lower_xpoint_fromtarget"] = psel["inner_lower_xpoint"] - psel["inner_lower_target_guard"] + 1
        psel["inner_upper_xpoint_fromtarget"] = psel["inner_upper_target_guard"] - psel["inner_upper_xpoint"] + 2
        psel["outer_upper_xpoint_fromtarget"] = psel["outer_upper_xpoint"] - psel["outer_upper_target_guard"] + 2
        psel["outer_lower_xpoint_fromtarget"] = self.g["nx"] - psel["outer_lower_xpoint"] + 1
        self.psel = psel

        ## 2D selectors
        s = self.s = {}
        for location in psel.keys():
            s[location] = (psel[location], slice(None, None))
        # s["inner_lower_target"] = (1, slice(None,None))
        # s["inner_lower_target_guard"] = (0, slice(None,None))
        # s["inner_upper_target"] = (upper_break-2, slice(None,None))
        # s["inner_upper_target_guard"] = (upper_break-1, slice(None,None))
        # s["outer_upper_target"] = (upper_break+1, slice(None,None))
        # s["outer_upper_target_guard"] = (upper_break, slice(None,None))
        # s["outer_lower_target"] = (-2, slice(None,None))
        # s["outer_lower_target_guard"] = (-1, slice(None,None))
        
        # First SOL rings
        for name in ["outer", "outer_lower", "outer_upper", "inner", "inner_lower", "inner_upper"]:
            s[name] = self.make_custom_sol_ring(name, i = 0)
            
        self.get_species()
        self.derive_data()
        
    def get_species(self):
        """
        Extract species indices from the balance file species matrix
        """
        
        bal = self.bal
        df_species = pd.DataFrame()

        nchars = bal["species"].shape[0]
        nspecies = bal["species"].shape[1]

        for ispecies in range(nspecies):
            
            species = ""
            for ichar in range(nchars):
                string = str(bal["species"][ichar, ispecies])
                string = string.replace("b", "").replace("'","")
                species += string
                
            df_species.loc[ispecies, "name"] = species.strip()
            
        self.species = df_species
        
        
    def get_impurity_stats(self, species_name, all_states = False):
        """
        Get sum of radiation for all species containing the string "species_name"
        and put it in the balance file. Units are W/m3
        
        all_states: save densities for neutral state and all charged states
        
        """
        try:
            species = self.species
        except:
            print("Species not found, generating")
            self.get_species()
        bal = self.bal
        
        dcz = species[species["name"].str.contains(fr"{species_name}\+")]
        species_indices = list(dcz.index)
        
        if len(species_indices) < 1:
            raise ValueError(f"No species found matching {species_name}")

        rad = []
        nimp = []
        for i in species_indices:
            rad.append(bal["b2stel_she_bal"][:,:,i])
            nimp.append(bal["na"][:,:,i])

        rad = np.sum(rad, axis = 0)
        nimp = np.sum(nimp, axis = 0)
        
        self.bal[f"R{species_name}"] = abs(rad / bal["vol"] )
        self.bal[f"n{species_name}"] = nimp
        self.bal[f"f{species_name}"] = nimp / self.bal["ne"]
        self.bal[f"f{species_name}tot"] = nimp / (self.bal["ne"] + self.bal["Na"] + (self.bal["Nm"]*2))
        
        if all_states:
            print("Saving all states")
            dcz = species[species["name"].str.contains(f"{species_name}")]
            for i in dcz.index:
                species = dcz.loc[i, "name"]
                self.bal[f"n{species}"] = bal["na"][:,:,i]
            
        
        print(f"Added total radiation, density and fraction for {species_name}")
        
    
    def derive_data(self):
        """
        Add new derived variables to the balance file
        This includes duplicates which are in Hermes-3 convention
        """
        
        bal = self.bal
        g = self.g
        
        ## EIRENE derived variables have extra values in the arrays
        ## Either 5 or 2 extra zeros at the end in poloidal depending on topology
        double_null = True
        if double_null is True:
            ignore_idx = 5
        else:
            ignore_idx = 2
        
        qe = constants("q_e")
        Mi = constants("mass_p") * 2

        bal["Td+"] = bal["ti"] / qe
        bal["Te"] = bal["te"] / qe
        bal["Ne"] = bal["ne"]
        bal["Pe"] = bal["ne"] * bal["Te"] * qe
        bal["Pd+"] = bal["ne"] * bal["Td+"] * qe

        # First index in third dimension is always main ion neutral
        bal["Na"] = bal["dab2"][:-ignore_idx, :, 0]
        bal["Nm"] = bal["dmb2"][:-ignore_idx, :, 0]
        bal["Nn"] = bal["Na"] + bal["Nm"] * 2
        bal["Ta"] = (bal["tab2"][:-ignore_idx, :, 0] / qe)
        bal["Td"] = bal["Ta"]
        bal["Tm"] = (bal["tmb2"][:-ignore_idx, :, 0] / qe)
        bal["Pa"] = bal["Na"].squeeze() * bal["Ta"].squeeze() * qe
        bal["Pm"] = bal["Nm"].squeeze() * bal["Tm"].squeeze() * qe

        bal["Pn"] = bal["Pa"] + bal["Pm"]
        bal["Tn"] = bal["Pn"] / bal["Nn"] / qe
        
        # Derive total fluxes (excl. drifts)
        bal["fhe_total"] = bal["fhe_cond"] + 5/3*bal["fhe_32"] + bal["fhe_thermj"]
        bal["fhi_total"] = bal["fhi_cond"] + 5/3*bal["fhi_32"]
        
        # Split fluxes into X and Y components
        existing_params = list(bal.keys())
        list_species = self.species["name"].values
        
        # NOTE: check that this works if you have multiple ions.
        # It's possible that all the arrays have 4 dimensions then
        for param in existing_params:
            for prefix in ["fhe", "fhi", "fmo", "fna"]:
                if prefix in param and not any([name in param for name in ["x", "y"]]):
                    
                    # Direction split only
                    if len(bal[param].shape) == 3:
                        bal[param.replace(prefix, f"{prefix}x")] = bal[param][:,:,0]
                        bal[param.replace(prefix, f"{prefix}y")] = bal[param][:,:,1]
                    
                    # Species split
                    elif len(bal[param].shape) == 4:
                        for i, species in enumerate(list_species):
                            bal[param.replace(prefix, f"{prefix}x_{species}")] = bal[param][:,:,0,i]
                            bal[param.replace(prefix, f"{prefix}y_{species}")] = bal[param][:,:,1,i]
                        
                    
        
        bal["hpar"] = g["hx"] * abs(g["Btot"] / abs(g["Bpol"]))   # Parallel cell width [m]
        bal["apar"] = bal["vol"] / bal["hpar"]            # Parallel cross-sectional area [m2]
                    
        bal["fhx_total"] = bal["fhex_total"] + bal["fhix_total"]
        bal["fhy_total"] = bal["fhey_total"] + bal["fhiy_total"]
        
        # flux = bal["fnax_D+1_tot"] / (bal["vol"] / bal["
        
        bal["Vd+"] = bal["ua"][:,:,1]  # Note first species is fluid neutral
        bal["NVd+"] = bal["Vd+"] * bal["Ne"] * Mi  # Parallel momentum flux kg/m2/s
        bal["M"] = np.abs(bal["Vd+"] / np.sqrt((bal["Te"]*qe + bal["Td+"]*qe)/Mi))  # Mach number
        
        ## NOW FOUND ua, DONT NEED THIS. besides fluxes are at cell edges
        # if "fnax_D+1_tot" in bal:
        #     bal["NVd+"] = bal["fnax_D+1_tot"] * Mi  / bal["apar"]  # Parallel momentum flux kg/m2/s
        #     bal["Vd+"] = bal["NVd+"] / bal["Ne"] / Mi   # NOTE: this will break for multiple ions
        #     bal["M"] = bal["Vd+"] / np.sqrt((bal["Te"]*qe + bal["Td+"]*qe)/Mi)  # Mach number
        # else:
        #     print("fnax_tot not found, skipping momentum calculation")
            
        # Reaction channels
        # bal["Rd+_atm"] = bal["b2stel_she_bal"][:,:,1] / bal["vol"]  # Total atom radiation [W/m3]
        
        # We follow Hermes-3 convention: all channels as sources (per volume basis)
        # S is particle source, E energy transfer, R radiation (or system loss), F momentum transfer
        bal["Rd+_exiz"] = bal["eirene_mc_eael_she_bal"].sum(axis = 2) / bal["vol"]  # Loss of energy due to excitation and 13.6 iz potential [W/m3]
        bal["Rd+_mol"] = bal["eirene_mc_emel_she_bal"].sum(axis = 2) / bal["vol"]  # Molecule radiation summed over all strata [W/m3]
            
        bal["Sd+_iz"] = bal["eirene_mc_papl_sna_bal"][:,:,1,:].sum(axis = 2) / bal["vol"]  # Choose ion species and sum over strata
        bal["Sd+_rec"] = bal["eirene_mc_pppl_sna_bal"][:,:,1,:].sum(axis = 2) / bal["vol"]   # Choose ion species and sum over strata
        
                
        self.bal = bal
        self.params = list(self.bal.keys())


        
    def make_custom_sol_ring(self, name, i = None, sep_dist = None):
        """
        Inputs
        ------
            name, str:
                "outer", "outer_lower", "outer_upper", "inner", "inner_lower", "inner_upper"
                "inner_upper_leg", "outer_upper_leg", "inner_lower_leg", "outer_lower_leg"
            i, int: 
                SOL ring index (0 = first inside SOL) 
            sep_dist, int
                Separatrix distance in [m]
                
        Returns:
        ------
            selector tuple as (X,Y)
            
        NOTE: INCLUDES GUARD CELLS apart from legs....
        NOTE: DOES NOT SUPPORT DISCONNECTED DOUBLE NULL
        """
        
        if i != None and sep_dist == None:
            yid = self.g["sep"] + i
        if sep_dist != None and i == None:
            yid = np.argmin(np.abs(self.g["radial_dist"] - sep_dist))
            if yid < self.g["sep"]:
                yid = self.g["sep"]
                print("SOL ring would have been inside the core, forcing to sep")
        if i != None and sep_dist != None:
            raise ValueError("Use i or sep_dist but not both")
        if i == None and sep_dist == None:
            raise ValueError("Provide either i or sep_dist")
        
        ## Note: all of these start beyond the midplane so that you can interpolate to Z=0 later
        selections = {}
        selections["outer"] =       (slice(self.g["upper_break"],None), yid)
        selections["outer_lower"] = (slice(self.g["omp"]+1-1,None), yid)
        selections["outer_upper"] = (slice(self.g["upper_break"], self.g["omp"]+1+1), yid)
        
        for leg_name in ["inner_upper_leg", "outer_upper_leg", "inner_lower_leg", "outer_lower_leg"]:
            selections[leg_name] = (self.psel[leg_name], yid)
        
        
        selections["inner"] =       (slice(1, self.g["upper_break"]), yid)
        selections["inner_lower"] = (slice(1, self.g["imp"]+1), yid)
        selections["inner_upper"] = (slice(self.g["imp"]-1, self.g["upper_break"]), yid)

        
        if name in selections.keys():
            return selections[name]
        else:
            raise Exception(f"Region {name} not supported. \n Try outer, outer_lower, outer_upper, inner, inner_lower, inner_upper")
    

        
    def close(self):
        self.bal.close()
        
    def plot_2d(self, param,
             ax = None,
             norm = None,
             data = np.array([]),
             cmap = "Spectral_r",
             custom_cmap = None,
             antialias = True,
             linecolor = "k",
             linewidth = 0,
             vmin = None,
             vmax = None,
             logscale = False,
             linthresh = None,
             absolute = False,
             alpha = 1,
             separatrix = True,
             separatrix_kwargs = {},
             grid_only = False,
             cbar = True,
             axis_labels = True,
             dpi = 150,
             xlim = (None, None),
             ylim = (None, None)
    ):  
        # print(data)
        # print(len(data))
        # print(data.size)
        # print(type(data))
        
        if custom_cmap != None:
            cmap = custom_cmap

        if type(data) is type(np.array([])):
            # print("Data is numpy array")
            if not data.size > 0:
                data = self.bal[param]
        # else:
            # print(f"Data is not numpy array, it's {type(data)}")
        # elif data == None:
        #     data = self.bal[param]
        
        if grid_only is True:
            data = np.zeros_like(self.bal["te"])
        
        if absolute:
            data = np.abs(data)
        
        if vmin == None:
            vmin = data.min()
        if vmax == None:
            vmax = data.max()
        if norm == None:
            norm = create_norm(logscale, norm, vmin, vmax, linthresh = linthresh)
        if ax == None:
            fig, ax = plt.subplots(dpi = dpi)
        else:
            fig = ax.get_figure()
        

        # Following SOLPS convention: X poloidal, Y radial
        crx = self.bal["crx"]
        cry = self.bal["cry"]
        nx, ny = self.g["nx"], self.g["ny"]
        
        
        
        
        # In hermes-3 and needed for plot: lower left, lower right, upper right, upper left, lower left
        # SOLPS crx structure: lower left, lower right, upper left, upper right
        # So translating crx is gonna be 0, 1, 3, 2, 0
        # crx is [corner, Y(radial), X(poloidal)]
        idx = [np.array([0, 1, 3, 2, 0])]

        # Make polygons
        patches = []
        for i in range(nx):
            for j in range(ny):
                p = mpl.patches.Polygon(
                    np.concatenate([crx[i,j,:][tuple(idx)], cry[i,j,:][tuple(idx)]]).reshape(2,5).T,
                    
                    fill=False,
                    closed=True,
                )
                patches.append(p)
                
        # Polygon colors
        colors = data.flatten()
        
        if grid_only is True:
            polys = mpl.collections.PatchCollection(
                patches, alpha = alpha, norm = norm, cmap = cmap, 
                antialiaseds = antialias,
                edgecolors = linecolor,
                facecolor = "white",
                linewidths = linewidth,
                joinstyle = "bevel")
        
        else:
            polys = mpl.collections.PatchCollection(
                patches, alpha = alpha, norm = norm, cmap = cmap, 
                antialiaseds = antialias,
                edgecolors = linecolor,
                linewidths = linewidth,
                joinstyle = "bevel")
            polys.set_array(colors)
        
        ## Cbar
        if fig != None and grid_only == False and cbar == True:
            # From https://joseph-long.com/writing/colorbars/
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(polys, cax = cax)
            cax.grid(visible = False)
        ax.add_collection(polys)
        ax.set_aspect("equal")
        
        ## Somehow autoscale breaks sometimes
        xmin, xmax = crx.min(), crx.max()
        ymin, ymax = cry.min(), cry.max()
        xspan = xmax - xmin
        yspan = ymax - ymin

        ax.set_xlim(xmin - xspan*0.05, xmax + xspan*0.05)
        ax.set_ylim(ymin - yspan*0.05, ymax + yspan*0.05)
        
        if axis_labels is True:
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
        if grid_only is False:
            ax.set_title(param)
        
        if separatrix is True:
            self.plot_separatrix(ax = ax, **separatrix_kwargs)
            
        if xlim != (None, None):
            ax.set_xlim(xlim)
        if ylim != (None, None):
            ax.set_ylim(ylim)
            
    def plot_neutral_vectors(self,
                             ax, 
                             normalise_arrows = True,
                             flux = False,  # Plot neutral flux instead of speed?
                             vmin = None,
                             vmax = None,
                             cmap = "jet",
                             logscale = False,
                             width = 0.0025,
                             scale_mult = 1,
                             gridcolor = "lightgrey",
                             gridwidth = 0.5):
        
        fort46_path = os.path.join(self.path, "fort.46.pkl")
        with open(fort46_path, "rb") as f:
            f46 = pickle.load(f)
            
        triangles_path = os.path.join(self.path, "triangle_mesh.pkl")
        with open(triangles_path, "rb") as f:
            triangles = pickle.load(f)
            
            
        if ax == None:
            fig, ax = plt.subplots()

        nodes = triangles["nodes"]
        cells = triangles["cells"]
        triang = mpl.tri.Triangulation(nodes[:,0], nodes[:,1], cells)

        if flux:
            U = f46["vxdena"] / (2*constants("mass_p"))
            V = f46["vydena"] / (2*constants("mass_p"))
            clabel = "Flux [$m^2/s$]"
        else:
            U = f46["vxdena"] / f46["pdena"] / (2*constants("mass_p"))
            V = f46["vydena"] / f46["pdena"] / (2*constants("mass_p"))
            clabel = "Flux [$m/s$]"
            
        speed = np.hypot(U, V)

        tri_coords = nodes[cells]
        centroids = tri_coords.mean(axis=1)
        px = centroids[:, 0]
        py = centroids[:, 1]

        # normalize vectors to unit length for plotting
        # avoid division by zero
        scale = 5e4 * scale_mult
        if normalise_arrows:
            eps = 1e-16
            U = U / (speed + eps) * scale
            V = V / (speed + eps) * scale
            
        # set up log normalization (avoid zeros by clipping)
        if vmin is None:
            vmin = max(speed.min(), 1e-3)  # lower bound >0
        if vmax is None:
            vmax = speed.max()
            
        if logscale:
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            cbar_formatter = mpl.ticker.LogFormatterSciNotation(base=10)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar_formatter = None

        if ax is None:
            fig, ax = plt.subplots(figsize = (5,15))

        ax.triplot(triang, color = gridcolor, lw = gridwidth)


        Q = ax.quiver(px, py, U, V, speed,
                        angles='xy',        # no automatic rotation
                        scale_units='xy',   # scale in data units
                        cmap = cmap,
                        norm = norm,
                        scale=3e6,            # if vectors are already in desired length
                        width=width        # adjust arrow thickness
                        )

        # create a colorbar whose height matches ax
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(Q, cax=cax, format=cbar_formatter)
        cbar.set_label(clabel)
        cbar.ax.yaxis.set_tick_params(which='both', length=4)

        ax.set_aspect("equal")
        
            
            
    def plot_separatrix(self, ax, **separatrix_kwargs):
        
        kwargs = {**{"c" : "white", "ls" : "-"}, **separatrix_kwargs}

        R = self.g["crx"][:,:,0]
        Z = self.g["cry"][:,:,0]
        ax.plot(R[self.s["inner"]], Z[self.s["inner"]], **kwargs)
        ax.plot(R[self.s["outer"]], Z[self.s["outer"]], **kwargs)
        
    def get_1d_radial_data(
        self,
        params,
        region = "omp",
        verbose = False,
        guards = False,
        keep_geometry = False,
        interpolate = True
    ):
        """
        Returns OMP, IMP or target data from the balance file
        Note that the balance file shape is a transpose of the geometry shape (e.g. crx)
        and contains guard cells.        
        Target data is obtained from the guard cells which are very close to the wall.
        """
        
        bal = self.bal

        if type(params) == str:
            params = [params]
        
        # Poloidal selector
        if any([region in name for name in [
            "omp", "imp"]]):

            p = self.s[region] 
            selector = self.s[region]
            
        elif any([region in name for name in [
            "outer_lower_target", "inner_lower_target",
            "outer_upper_target", "inner_upper_target"]]):
            
            selector = self.s[region+"_guard"] # For targets take guard cell which is tiny & close to wall value
            
        else:
            raise Exception(f"Unrecognised region: {region}")
        
        ## Interpolate to Z = 0 for OMP and IMP
        if (region == "omp" or region == "imp") and interpolate == True:
            ## Select a couple of cells on each side of the midplane
            if region == "omp":
                pol_selector = slice(self.psel["omp"]-3, self.psel["omp"]+5)
            elif region == "imp":
                pol_selector = slice(self.psel["imp"]-4, self.psel["imp"]+4)
                

            no_rings = self.g["ny"]   # Number of radial cells
                
            # For every parameter, collect the value interpolated at Z = 0 

            df_pol_slice = pd.DataFrame()  # Dataframe containing 2D data around the midplane for 
            df_rad_slice = pd.DataFrame()   # Radial profile dataframe to construct

            for ring_id in range(no_rings):

                Z = self.g["Z"][pol_selector, ring_id]
                
                # Get parameters
                for param in ["Z", "R", "hx", "hy", "vol", "Btot", "Bpol"] + params:
                    if param in bal:
                        df_pol_slice[param] = bal[param][pol_selector, ring_id]
                    else:
                        print(f"Parameter {param} not found")
                    
                # Interpolate to Z=0
                
                for param in df_pol_slice.drop(["Z"], axis = 1).columns:
                    df_rad_slice.loc[ring_id, param] = scipy.interpolate.interp1d(df_pol_slice["Z"], df_pol_slice[param], kind = "cubic")(0)
                    
            df = df_rad_slice.copy()
            
        else:
            df = pd.DataFrame()
            for param in ["hx", "hy", "R", "Z", "hx", "Btot", "Bpol", "vol"] + params:
                df[param] = bal[param][selector]
                
        
        # Interpolate between cells to get separatrix distance
        hy = df["hy"].values
        vol = df["vol"].values
        
        for i, _ in enumerate(df["hy"]):
            if i == 0:
                df.loc[i, "dist"] = hy[i] / 2
            else:
                df.loc[i, "dist"] = df.loc[i-1, "dist"] + hy[i-1]/2 + hy[i]/2
        
        sepind = self.g["sep"]
        dist_sep = df["dist"][sepind] - (df["dist"][sepind] - df["dist"][sepind-1]) / 2
        df["dist"] -= dist_sep 
        df["sep"] = 0
        df.loc[sepind, "sep"] = 1

        
        dSpar = df["hx"] * abs(df["Btot"] / df["Bpol"])
        df["apar"] = df["vol"] / dSpar
        
        df.insert(0, "dist", df.pop("dist"))
        df.insert(1, "sep", df.pop("sep"))
        df.insert(2, "apar", df.pop("apar"))
        df.insert(3, "vol", df.pop("vol"))
        
        if keep_geometry is False:
            df = df.drop(["R", "hx", "hy", "Btot", "Bpol"], axis = 1)
        
        if guards is False:
            df = df.iloc[1:-1]  # Trim guards but keep indices          
        
        return df
    
    def get_1d_poloidal_data(
        self,
        params,
        region = "outer_lower",
        sepadd = None,
        sepdist = None,
        target_first = False,
        guards = False
        
    ):
        """
        Returns field line data from the balance file
        Note that the balance file shape is a transpose of the geometry shape (e.g. crx)
        and contains guard cells, so R and Z are incorrect because they don't have guards
        R and Z are provided for checking field line location only.
        only outer_lower region is provided at the moment.
        sepadd is the ring index number with sepadd = 0 being the separatrix
        TODO: Try using "conn" which is the connection length. There is a difference to the DLS due to interpolation
        NOTE: Be very careful when flipping the direction of the field line (target_first) as there are some hysteresis
        effects. Do all operations on Spar before reversing.
        """
        
        if type(params) == str:
            params = [params]
            
        if sepadd == None and sepdist == None:
            raise ValueError("Must use either index i or separatrix distance sepdist")
        
        elif sepadd != None and sepdist != None:
            raise ValueError("Must use either index i or separatrix distance sepdist, not both")
        
        elif sepdist != None:
            radial_df = self.get_1d_radial_data([], region = "omp")
            sepind = radial_df[radial_df["sep"] == 1].index[0]
            sepadd = radial_df.loc[(radial_df["dist"] - sepdist).abs().idxmin()].name - sepind
            
        
        yind = self.g["sep"] + sepadd   # Ring index
        omp = self.g["omp"]
        imp = self.g["imp"]
        
        selector = self.make_custom_sol_ring(region, i = sepadd)
        
        hx = self.bal["hx"][selector]
        Btot = self.g["Btot"][selector]
        Bpol = self.g["Bpol"][selector]
        R = self.g["R"][selector]
        Z = self.g["Z"][selector]
        vol = self.g["vol"][selector]
        
        data = {}
        
        for param in params:
            
            # Look in bal or geometry
            if param in self.bal:
                data[param] = self.bal[param]
            elif param in self.g:
                data[param] = self.g[param]
            else:
                raise ValueError(f"Parameter {param} not found")
            
            if param.startswith("fh") and "x" not in param and "y" not in param:
                data[param] = data[param][:,:,0]   # Select poloidal
            
            # Catch special variables with more dimensions
            if len(data[param].shape) > 2:
                raise ValueError(f"Paramerer {param} has more than 2 dimensions")
            
            
        
        df = pd.DataFrame()
        df["R"] = R
        df["Z"] = Z
        
        ## Poloidal connection length
        df["Spol"] = np.cumsum(hx)  # Poloidal distance
        
        # df["Spol"] -= df["Spol"].iloc[0]   # Now 0 is cell centre before midplane
        
        for param in params:
            df[param] = data[param][selector]
        
        ## Parallel connection length
        dSpar = hx * abs(Btot / abs(Bpol))
        df["Spar"] = np.cumsum(dSpar)
        df["apar"] = vol / dSpar


        if any([name in region for name in ["outer_upper", "inner_lower"]]):
            df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
            df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
            df = df.iloc[::-1].reset_index(drop = True)

        # Interpolate onto Z = 0
        for param in df.columns.drop("Z"):
            interp = scipy.interpolate.interp1d(df["Z"], df[param], kind = "linear")
            df.loc[0, param] = interp(0)
        df.loc[0,"Z"] = 0  

        df["Spol"] -= df["Spol"].iloc[0]   # Now 0 is at Z = 0
        df["Spar"] -= df["Spar"].iloc[0]   # Now 0 is at Z = 0
        
        
        ## Value of X-point column will be 1 in the cell just downstream of X-point
        # df["Xpoint"] = ""
        Xpoint_index = df.index[-self.psel[f"{region}_xpoint_fromtarget"]+1]
        
        df.loc[:Xpoint_index, "region"] = "upstream"
        df.loc[Xpoint_index:, "region"] = "divertor"
        
        df["Xpoint"] = 0
        df.loc[Xpoint_index, "Xpoint"] = 1
        
        # df.loc[Xpoint_index + 1, "Xpoint"] = "before"
        # df.loc[Xpoint_index, "Xpoint"] = "after"
        # df["Xpoint"] = 1


        if target_first:
            df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
            df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
            df = df.iloc[::-1]
        
        df = df.reset_index(drop = True)
        
        return df
        # return df[::-1]
    
    def find_param(self, name):
        """
        Returns variables that match string
        """
        for param in self.params:
            if name in param: print(param)
            
    # def find_xpoints(self, plot = False):
    #     """
    #     Returns poloidal IDs of xpoints, works for double null only
    #     the IDs are for the cell immediately after the X-point
    #     in the direction of ascending X (poloidal) coordinate.
    #     They are arranged clockwise starting at inner lower
        
    #     NOT VERY ROBUST
        
    #     DEPRECATED - USING DAVID MOULTON'S XCUT METHOD INSTEAD
    #     """

        # x = range(self.g["nx"])
        # R = self.g["R"][:,self.g["sep"]]

        # inner_R = self.g["R"][slice(None, self.g["upper_break"]),self.g["sep"]]
        # inner_x = np.array(range(self.g["upper_break"]))


        # outer_R = self.g["R"][slice(self.g["upper_break"], None),self.g["sep"]]
        # outer_x = np.array(range(self.g["upper_break"], self.g["nx"]))

        # import scipy

        # inner_ids = scipy.signal.find_peaks(inner_R, threshold = 0.0007)
        # outer_ids = scipy.signal.find_peaks(outer_R*-1, threshold = 0.001)

            
                
        # issue = False
        
        # if (len(inner_ids[0]) > 2):
        #     print("Too many X-points found in inner SOL")
        #     issue = True
        # elif (len(inner_ids[0]) < 2):
        #     print("Too few X-points found in inner SOL")
        #     issue = True
            
        # if (len(outer_ids[0]) > 2):
        #     print("Too many X-points found in outer SOL")
        #     issue = True
            
        # elif (len(outer_ids[0]) < 2):
        #     print("Too few X-points found in outer SOL")
        #     issue = True
            
        # if issue or plot:
        #     fig, axes = plt.subplots(1,2, figsize = (8,4))
        #     ax = axes[0]
        #     ax.set_title("Inner SOL")
        #     ax.plot(inner_x, inner_R)
        #     for i in inner_ids[0]:
        #         ax.scatter(inner_x[i], inner_R[i])
        #     ax = axes[1]
        #     ax.set_title("Outer SOL")
        #     ax.plot(outer_x, outer_R)
        #     for i in outer_ids[0]:
        #         ax.scatter(outer_x[i], outer_R[i])
                
        # if issue:
        #     raise Exception("Issue in peak finder, try reducing threshold")

        # xpoints = [
        #     inner_ids[0][0], 
        #     inner_ids[0][1]+1, 
        #     outer_ids[0][0]+ self.g["upper_break"] + 1, 
        #     outer_ids[0][1]+ self.g["upper_break"]
        #     ]

        # return xpoints
    
    def plot_selection(self, sel, ax = None, **kwargs):
        """
        Plots scatter of selected points according to the tuple sel
        which contains slices in (X, Y)
        Examples provided in self.s
        """
        
        data = self.bal["ne"][:]
        data = np.zeros_like(data)

        data[sel] = 1
        
        def_kwargs = dict(
            separatrix_kwargs = dict(lw = 0.5, ls = "--", c = "white"),
            linewidth = 0.1)
        
        plot_kwargs = {**def_kwargs, **kwargs}
        
        if "dpi" in kwargs:
            dpi = kwargs["dpi"]
        else:
            dpi = 150
            
        if ax == None:
            fig, ax = plt.subplots(dpi = dpi)

        self.plot_2d("", ax = ax, data = data, cmap = mpl.colors.ListedColormap(["lightgrey", "limegreen"]),
             antialias = True, 
             cbar = False,
             **plot_kwargs)
        
        ax.scatter(self.g["R"][sel], self.g["Z"][sel], c = "deeppink", s = 3)
        
    def extract_cooling_curve(self, species, region, sepadd, order = 10, 
                              plot = False, cz_effect = False,
                              ne_effect = False, other_losses = False, constant_cz = 0.05,
                              polyval = True):
        """
        Extract cooling curve from a single SOL ring
        Inputs: species (e.g. Ar). Must calculate RAr and fAr first (get radiation stats)
        region: string  like "outer_lower"
        sepadd: no. sol rings beyond separatrix
        order: order of polynomial fit
        plot: debug plot
        
        Returns a function for the cooling curve which takes a float
        and returns the curve in Wm3
        """
        
        if other_losses:
            param_list = ["Te", f"R{species}", "RC", "Rd+_exiz", "Rd+_mol", f"f{species}", "ne"]
        else:
            param_list = ["Te", f"R{species}", f"f{species}", "ne"]

        solps = self.get_1d_poloidal_data(param_list, sepadd = sepadd, region = region)
        solps = solps.iloc[:-1]  # Ignore guard cell
        solps[f"R{species}"] = abs(solps[f"R{species}"])

        x = solps["Te"]
        upper_limit = solps["Te"].max()  # Upper validity limit
        
            
        if other_losses:
            solps["RC"] = np.abs(solps["RC"])
            solps["Rd+_exiz"] = np.abs(solps["Rd+_exiz"])
            solps["Rd+_mol"] = np.abs(solps["Rd+_mol"])
            Qrad = solps[f"R{species}"] + solps["RC"] + solps["Rd+_exiz"] + solps["Rd+_mol"]
        else:
            Qrad = solps[f"R{species}"]
            
        if cz_effect:
            cz = constant_cz
        else:
            cz = solps[f"f{species}"]
            
        if ne_effect:
            upstream = solps.iloc[0]
            ne = upstream["ne"] * upstream["Te"] / solps["Te"]
        else:
            ne = solps["ne"]
            
        y = Qrad/(ne**2 * cz)
        
        
        if polyval:
            logx = np.log10(x)
            logy = np.log10(y)

            coeffs = np.polyfit(logx, logy, order)
            fit_logy = np.polyval(coeffs, logx)
            
            
            if plot:
                ms = 2
                fig, axes = plt.subplots(1,2, figsize = (10,4))
                axes[0].plot(logx, logy, c = "k", marker = "o", lw = 0, ms = ms, label = "SOLPS")
                
                axes[0].set_title("Log space")
                axes[0].set_xlabel("logTe")
                axes[0].set_ylabel("logLz")
                
                axes[1].plot(x, 10**(fit_logy), c = "darkorange")
                axes[1].plot(x, y, c = "k", marker = "o", lw = 0, ms = ms, label = "SOLPS")
                axes[1].set_title("Linear space")
                axes[1].set_xlabel("Te")
                axes[1].set_ylabel("Lz")
                
                for ax in axes:
                    ax.legend()
                
                
            
            def fit(T):
            
                if not any([type(T) != format for format in [np.float64, float]]):
                    raise ValueError("T must be a float")

                logT = np.log10(T)
                fit = lambda x: 10 ** np.polyval(coeffs, logT)
                
                if T > upper_limit:
                    return 0
                elif T < 2:
                    return 0
                else:
                    return fit(logT)
                
        else:
            interp = scipy.interpolate.interp1d(np.log10(x), y, kind = "quadratic")
            
            def fit(T):
                if not any([type(T) != format for format in [np.float64, float]]):
                    raise ValueError("T must be a float")

                if T > upper_limit:
                    return 0
                elif T < 2:
                    return 0
                else:
                    return interp(np.log10(T))
            
        return fit

    def extract_kappa0(self, sepadd, region, target_first = True, plot =False, 
                       qpar_weighted = True, total_hflux = True, skip_xpoint = False, print_kappa=False):
        
        df = self.get_1d_poloidal_data(["Te", "fhex_cond", "fhx_total"], 
                                  sepadd = sepadd, region = region, target_first = target_first)
                                  

        if "outer_upper" in region or "inner_lower" in region:
            mult = -1
        else:
            mult = 1
            
        if total_hflux:
            df["qpar"] = df["fhx_total"] / df["apar"] * mult
        else:
            df["qpar"] = df["fhex_cond"] / df["apar"] * mult

        ### TRIM FIELD LINE
        df = df.iloc[1:-1] # Remove guard cells
        # df = df[df["Te"]>10] # Discard below front (temp)
        qpar_front_idx = df[df["qpar"] > df["qpar"].max() * 0.15].index[0]  # Discard below front (qpar)
        df = df.iloc[qpar_front_idx:]
        df = df.iloc[:-2]

        # Skip points around X-point due to temperature gradient anomaly
        if skip_xpoint:
            xpoint_index = df[df["Xpoint"] == 1].index[0]
            skip_indices = [xpoint_index-1, xpoint_index, xpoint_index+1]
            df = df[~df.index.isin(skip_indices)]

        ### Kappa calc
        df["gradT"] = np.gradient(df["Te"], df["Spar"])
        df = df[df["gradT"]>0]  # Discard negative T gradient

        df["kappa0"] = df["qpar"] / (df["Te"]**(5/2) * df["gradT"]) 
        df["kappa0_times_qpar"] = df["kappa0"] * df["qpar"]

            
        if plot:
            
            dfx = df[df["Xpoint"]==1]
            fig, ax = plt.subplots()
            ax.plot(df["Spar"], df["kappa0"], marker = "o", label = "kappa0", ms = 3, lw = 1)
            ax.plot(dfx["Spar"], dfx["kappa0"], lw = 0, marker = "x", ms = 3)
            ax2 = ax.twinx()
            ax2.plot(df["Spar"], df["Te"], c = "r", marker = "o", label = "Te", ms = 3, lw = 1)
            
            ax.set_ylabel("kappa0")
            ax2.set_ylabel("Te [eV]")
            ax3 = ax.twinx()
            ax3.plot(df["Spar"], df["qpar"], c = "darkorange", marker = "*", label = "qpar")
            
            for x in [ax, ax2, ax3]:
                x.grid(which = "both", visible = False)
            fig.legend(loc = "upper right", bbox_to_anchor = (0.9, 0.9))
            
        if qpar_weighted:
            out = (df["kappa0_times_qpar"] / df["qpar"].sum()).sum()
        else:
            out = df["kappa0"].mean()
            
        if print_kappa:
            print(f"Kappa0: {out:.0f}")
            
        return out
    
    def extract_front_pos(self, sepadd, region, impurity = "N", method = "qpar_tot", threshold = 0.05):
        """
        Extract front position. 
        
        Inputs
        ------
        sepadd: no. of SOL rings beyond separatrix
        region: string like "outer_lower"
        impurity: string like "N"
        method: usually qpar_tot
        threshold: fraction of max value to define front
        """

        df = self.get_1d_poloidal_data(["Te", "fhex_cond", "fhx_total", f"R{impurity}", "Btot"], 
                                  sepadd = sepadd, region = region, target_first = True)
        

        df["qpar_cond"] = df["fhex_cond"] / df["apar"]
        df["qpar_tot"] = df["fhx_total"] / df["apar"]
        df["qpar_total"] = df["qpar_tot"]

        param = method
        
        if "qpar" in param:
            df[param] = df[param].abs()
            max = df[param].max()
            max_idx = df[param].idxmax()
            df_low = df.loc[:max_idx]
            df_low_rev = df_low.iloc[::-1].reset_index(drop=True)

            df_after_front = df_low_rev[df_low_rev[param] <= max*threshold].reset_index(drop = True)
            if len(df_after_front) == 0:
                front_spar = 0
            else:
                front_spar = df_after_front.iloc[0]["Spar"]
            
        else:
            cumR = scipy.integrate.cumulative_trapezoid(y = df[f"R{impurity}"] / df["Btot"], x = df["Spar"], initial = 0)
            df["cumR"] = cumR/cumR[-1]
            df_after_front = df[df[param] <= threshold].reset_index(drop = True)
            front_spar = df_after_front.iloc[-1]["Spar"]
        
        
        return front_spar
    
    
    def get_leg_energy_balance(self, sepadd, region, impurities = ["N", "C"], plot = False):
        
        impurity_variables = [f"R{impurity}" for impurity in impurities]
        
        fluxlist = ["fhex_total", "fhey_total", "fhix_total", "fhiy_total", "fhey_32", "fhey_cond", "fhiy_32", "fhiy_cond", "fhx_total"]
        fline = self.get_1d_poloidal_data(impurity_variables + fluxlist + ["vol","Rd+_exiz", "Rd+_mol"], sepadd = sepadd, region = region, target_first = True)
        fline_right = self.get_1d_poloidal_data(fluxlist, sepadd = sepadd+1, region = region, target_first = True)
        
        for param in fluxlist:
            if "inner_lower" in region or "outer_upper" in region:
                fline[param] *= -1
                fline_right[param] *= -1
                
        if fline["fhx_total"].max() < 0:
            raise Exception("Total heat flux negative: fix sign")
        
        fline_x = fline.query("Xpoint == 1").squeeze()  # First cell after X-point
        fline_right_x = fline_right.query("Xpoint == 1").squeeze()
        fline_div = fline.query("region == 'divertor'")   # Cells below X-point
        fline_right_div = fline_right.query("region == 'divertor'") 
        out = {}
        for impurity in impurities:
            out[f"P_{impurity}"] = (fline[f"R{impurity}"] * fline["vol"]).sum()
            
        
            
        
        out["P_H"] = ((fline["Rd+_exiz"].abs() + fline["Rd+_mol"].abs()) * fline["vol"]).sum()
        out["P_div"] = fline_x["fhex_total"] + fline_x["fhix_total"]
        
        out["qpar_mapped_to_target"] = out["P_div"] / fline.iloc[0]["apar"]
        
        # Radial divergences: positive = flow radially outwards
        # Top = towards SOL, bottom = towards core (logical grid convention)
        # out["P_radial_e_bottom"] = fline_div["fhey_total"].sum()
        # out["P_radial_e_top"] = fline_right_div["fhey_total"].sum() 
        
        # out["P_radial_i_bottom"] = fline_div["fhiy_total"].sum() 
        # out["P_radial_i_top"] = fline_right_div["fhiy_total"].sum()  
        
        # Need to know what "stays" in the tube, so take what entered and subtract what left
        # out["P_radial_e"] = (out["P_radial_e_bottom"] - out["P_radial_e_top"]).sum() 
        # out["P_radial_i"] = (out["P_radial_i_bottom"] - out["P_radial_i_top"]).sum() 
        # out["P_radial"] = out["P_radial_e"] + out["P_radial_i"]
        
        P_radial_e_bottom = fline_div["fhey_total"]
        P_radial_e_top = fline_right_div["fhey_total"] 
        
        P_radial_i_bottom = fline_div["fhiy_total"] 
        P_radial_i_top = fline_right_div["fhiy_total"]  
        
        P_radial_e = P_radial_e_bottom - P_radial_e_top
        P_radial_i = P_radial_i_bottom - P_radial_i_top
        P_radial = P_radial_e + P_radial_i
        
        out["P_radial_e"] = P_radial_e.sum() 
        out["P_radial_i"] = P_radial_i.sum() 
        out["P_radial"] = P_radial.sum()
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(fline_div["Spar"], fline_div["fhx_total"], label = "total", c = "k")
            ax.plot(fline_div["Spar"], (fline_div["Rd+_exiz"].abs() + fline_div["Rd+_mol"].abs()) * fline_div["vol"], label = "H")
            ax.plot(fline_div["Spar"], fline_div["RAr"]*fline_div["vol"], label = "Ar")
            ax.plot(fline_div["Spar"], P_radial, label = "radial")
            ax.set_title(f"{region}, ring {sepadd+1}")
            ax.legend()
            ax.set_ylabel(f"Integrated power source [W]")
            ax.set_xlabel("Spar")


        return out
    
    def get_leg_particle_balance(self, sepadd, region, Te_threshold = 0):
        """
        Return integrals of particle sources along a single field line
        and comparison of them to the particle flux entering the divertor
        Ability to filter by temperature to exclude detachment region with Te_threshold
        """
        
        variables = ["Te", "vol", "fnax_D+1_pll", "fnay_D+1_nanom", "Sd+_iz", "Sd+_rec"]
        fline = self.get_1d_poloidal_data(variables, sepadd = sepadd, region = region, target_first = True)
        fline_right = self.get_1d_poloidal_data(variables, sepadd = sepadd+1, region = region, target_first = True)
        

        fline_x = fline.query("Xpoint == 1").squeeze()  # First cell after X-point
        fline_right_x = fline_right.query("Xpoint == 1").squeeze()
        fline_div = fline.query(f"region == 'divertor' & Te > {Te_threshold}")   # Cells below X-point
        fline_right_div = fline_right.query(f"region == 'divertor' & Spar > {fline_div['Spar'].min()}")  # Same filter on RHS fline   
        
        if any([x in region for x in ["inner_lower"]]):
            mult = -1
        else:
            mult = 1
            
        # plt.plot(fline_div["Spar"] , fline_div["Sd+_iz"] * fline_div["vol"])
        
        out = {}
        out["S_div"] = fline_x["fnax_D+1_pll"] * mult
        out["S_iz"] = (fline_div["Sd+_iz"]*fline_div["vol"]).sum()
        
        out["S_rec"] = (fline_div["Sd+_rec"]*fline_div["vol"]).sum()
        
        out["S_radial_bottom"] = fline_div["fnay_D+1_nanom"].sum() * -1 * mult
        out["S_radial_top"] = fline_right_div["fnay_D+1_nanom"].sum() * -1 * mult
        out["S_radial"] = out["S_radial_bottom"] - out["S_radial_top"]
        
        return out

                 

        
def create_norm(logscale, norm, vmin, vmax, linthresh = None):
    if logscale:
        if norm is not None:
            raise ValueError(
                "norm and logscale cannot both be passed at the same time."
            )
        if vmin * vmax > 0:
            # vmin and vmax have the same sign, so can use standard log-scale
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            # vmin and vmax have opposite signs, so use symmetrical logarithmic scale
            if not isinstance(logscale, bool):
                linear_scale = logscale
            else:
                linear_scale = 1.0e-5
            
            if linthresh is None:
                linear_threshold = min(abs(vmin), abs(vmax)) * linear_scale
                if linear_threshold == 0:
                    linear_threshold = 1e-4 * vmax   # prevents crash on "Linthresh must be positive"
            else:
                linear_threshold = linthresh
            norm = mpl.colors.SymLogNorm(linear_threshold, vmin=vmin, vmax=vmax)
    elif norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return norm

def read_display_tallies(path):
    """ 
    This reads a csv file coming from a copy paste of the output of the display_tallies SOLPS routine
    It parses the table into three dataframes corresponding to the volumetric tallies,
    and the tallies of fluxes between regions in the x and y directions
    Each variable comes in three flavours: suffixless = total, _n = neutrals, _i = ions
    """
    
    # Read display tallies csv copied from console
    df = pd.read_csv(path, skiprows=2, names = range(16), sep = r"\s+", index_col = 0)

    # Make a col "param" with parameters - rest of columns are the 0-14 regions (double null only)
    df = df[df.index.notnull()].reset_index()
    df.columns = df.columns - 1
    df = df.rename({-1 : "param"}, axis = 1)
    
    # Some lines have no parameter title and the delimiter parsing makes it start as a number
    # Shift those back to be in line with the others
    df[df["param"].str.contains(r"E\+")] = df[df["param"].str.contains(r"E\+")].shift(axis=1)

    params = df["param"][df["param"].str.contains(r"D0|D\+|None") == False].values

    for i in df.index:
        # Empty rows due to the shifting above are now labelled correctly
        if df.loc[i, "param"] == None:
            df.loc[i, "param"] = df.loc[i-1, "param"]
            df.loc[i-1, "param"] = None

        # Label D0 as _n, D+1 as _i and D0 + D+1 as the original parameter
        if df.loc[i, "param"] == "D0":
            name = df.loc[i-1, "param"]
            df.loc[i, "param"] = name+"_n"
            df.loc[i+1, "param"] = name+"_i"
            df.loc[i-1] = df.loc[i] + df.loc[i+1]   # The name without suffix becomes sum of neutrals and ions
            df.loc[i-1, "param"] = name
            
    # Remove all None junk
    df = df[~df["param"].isna()]

    for i in range(15):
        df[i] = df[i].astype(float)

    # Parse the volumetric regions
    dfreg = df[((df["param"].str.contains("x|y") == False) & (df["param"].str.contains("reg") == True))]
    dfreg = dfreg.rename({
        0 : "total",
        1 : "inner_core",
        2 : "inner_sol",
        3 : "inner_lower_target",
        4 : "inner_upper_target",
        5 : "outer_core",
        6 : "outer_sol",
        7 : "outer_upper_target",
        8 : "outer_lower_target"
    }, axis = 1
    ).dropna(axis=1)

    # Parse the fluxes between regions in the X direction
    dfxreg = df[((df["param"].str.contains("xreg") == True))]
    dfxreg = dfxreg.rename({
        0 : "total",
        1 : "inner_lower_target",
        2 : "inner_lower_entrance",
        3 : "inner_upper_entrance",
        4 : "inner_upper_target",
        5 : "outer_upper_target",
        6 : "outer_upper_entrance",
        7 : "outer_lower_entrance",
        8 : "outer_lower_target",
        9 : "lower_inner_pfr_to_inner_core",
        10 : "inner_core_to_inner_upper_pfr",
        11 : "outer_upper_pfr_to_outer_core",
        12 : "outer_core_to_outer_lower_pfr"
    }, axis = 1
    ).dropna(axis=1)
    
    # Parse the fluxes between regions in the Y direction
    dfyreg = df[((df["param"].str.contains("yreg") == True))]
    dfyreg = dfyreg.rename({
        0 : "total",
        1 : "inner_lower_pfr",
        2 : "inner_core",
        3 : "inner_upper_pfr",
        4 : "inner_separatrix",
        5 : "inner_lower_sol",
        6 : "inner_sol",
        7 : "inner_upper_sol",
        8 : "outer_upper_pfr",
        9 : "outer_core",
        10 : "outer_lower_pfr",
        11 : "outer_separatrix",
        12 : "outer_upper_sol",
        13 : "outer_sol",
        14 : "outer_lower_sol"
    }, axis = 1
    ).dropna(axis=1)
    
    for dfout in [dfreg, dfxreg, dfyreg]:
        dfout.index = dfout.pop("param")

    return {"reg":dfreg, "xreg":dfxreg, "yreg":dfyreg }

def extract_eirene_spectra(logpath):
    """
    Extracts EIRENE spectra saved to log file.
    WARNING: WILL ONLY GET THE FIRST INSTANCE!
    
    """
    ## Extract spectrum relevant lines
    lines = []
    read = False
    with open(logpath, 'r') as f:
        for line in f:
            
            if "THIS IS THE SUM OVER THE STRATA" in line:
                read = True
            if "B2-EIRENE GLOBAL BALANCES" in line:
                read = False
                
            if read:
                lines.append(line.rstrip())
                
    ## Parse for stats and locate data
    for i, line in enumerate(lines):
        if "NUMBER OF MONTE CARLO HISTORIES" in line:
            n_histories = float(lines[i+1].split()[-1])
        if "SPECTRUM CALCULATED FOR SCORING CELL" in line:
            cell_id = int(line.split()[-1])
        if "INTEGRAL OF SPECTRUM" in line:
            spectrum_integral = float(line.split()[-1])
        
        if "B-LEFT" in line:
            line_data_bin0 = i+1
            line_data_start = i+3
            
        if "TEST PARTICLE INFLUX FROM SOURCE (AMP)" in line:
            line_data_end = i-5
            line_data_lastbin = i-3
        
    ## Extract lines with spectrum data    
    spectrum_lines = []
    spectrum_lines.append(lines[line_data_bin0])
    for i in range(line_data_start, line_data_end):
        spectrum_lines.append(lines[i])
        
        
    ## Extract data from lines
    bin = []
    B_left = []
    B_right = []
    flux_per_bin = []

    for i, line in enumerate(spectrum_lines):
            
            line_split = line.strip().split(" ")
            bin.append(int(line_split[0]))
            B_left.append(float(line_split[2]))
            B_right.append(float(line_split[4]))
            flux_per_bin.append(float(line_split[6]))


    out = {}
    out["B_left"] = np.array(B_left)
    out["B_right"] = np.array(B_right)
    out["B_mean"] = (out["B_left"] + out["B_right"]) / 2
    out["flux_per_bin"] = np.array(flux_per_bin)
    out["bin"] = np.array(bin)
    out["n_histories"] = n_histories
    out["spectrum_integral"] = spectrum_integral
    out["cell_id"] = cell_id
    
    return out

def returnll(R, Z):
    # return the poloidal distances from the target for a given configuration
    # C.Cowley 2021
    PrevR = R[0]
    ll = []
    currentl = 0
    PrevZ = Z[0]
    for i in range(len(R)):
        dl = np.sqrt((PrevR - R[i]) ** 2 + (PrevZ - Z[i]) ** 2)
        currentl += dl
        ll.append(currentl)
        PrevR = R[i]
        PrevZ = Z[i]
    return ll


def returnS(R, Z, B, Bpol):
    # return the real total distances from the target for a given configuration
    # C.Cowley 2021
    PrevR = R[0]
    s = []
    currents = 0
    PrevZ = Z[0]
    for i in range(len(R)):
        dl = np.sqrt((PrevR - R[i]) ** 2 + (PrevZ - Z[i]) ** 2)
        ds = dl * np.abs(B[i]) / np.abs(Bpol[i])
        currents += ds
        s.append(currents)
        PrevR = R[i]
        PrevZ = Z[i]
    return s



