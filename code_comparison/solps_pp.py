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
        
        HEAT FLUXES
        ------------
        ALL ARE IN W
        
        The third dimension is the direction, 0 = poloidal, 1 = radial
        fhi_32 is the convective ion heat flux (52 is only if you use total energy)
        fhi_cond is conductive
        fhe_thermj is the parallel current electron heat flux
        Rest are to do with drifts or currents
        
        fht = total heat flux = fhe_cond + 5/3*fhe_32 + fhi_cond + 5/3*fhi_32 + fhe_thermj
        
        """
        
        raw_balance = nc.Dataset(os.path.join(path, "balance.nc"))
        
        ## Need to transpose to get over the backwards convention compared to MATLAB
        bal = {}
        for key in raw_balance.variables:
            bal[key] = raw_balance[key][:].transpose()
            
        self.bal = bal
        
        raw_balance.close()

        self.params = list(bal.keys())
        self.params.sort()
        
        # Set up geometry
        
        g = self.g = {}
        
        # Get cell centre coordinates
        R = self.g["R"] = np.mean(bal["crx"], axis=2)
        Z = self.g["Z"] = np.mean(bal["cry"], axis=2)
        self.g["hx"] = bal["hx"]
        self.g["hy"] = bal["hy"]
        self.g["crx"] = bal["crx"]
        self.g["cry"] = bal["cry"]
        self.g["nx"] = self.g["crx"].shape[0]
        self.g["ny"] = self.g["crx"].shape[1]
        self.g["vol"] = self.bal["vol"]
        
        self.g["Btot"] = bal["bb"][:,:,3]
        self.g["Bpol"] = bal["bb"][:,:,0]
        self.g["Btor"] = np.sqrt(self.g["Btot"]**2 - self.g["Bpol"]**2) 
        
        
        
        leftix = bal["leftix"]+1
        leftix_diff = np.diff(leftix[:,1])
        g["xcut"] = xcut = np.argwhere(leftix_diff<0).squeeze()
        g["leftcut"] = [xcut[0]-2, xcut[1]+2]
        g["rightcut"] = [xcut[4]-1, xcut[3]-1]
        
        omp = self.g["omp"] = int((g["rightcut"][0] + g["rightcut"][1])/2) + 1
        imp = self.g["imp"] = int((g["leftcut"][0] + g["leftcut"][1])/2)
        upper_break = self.g["upper_break"] = g["xcut"][2] + 1
        sep = self.g["sep"] = bal["jsep"][0] + 2
        # self.g["xpoints"] = self.find_xpoints()   # NOTE: Not robust
        
        
        ## Prepare selectors
        
        s = self.s = {}
        s["imp"] = (imp, slice(None,None))
        s["omp"] = (omp, slice(None,None))
        s["outer_lower_target"] = (-2, slice(None,None))
        s["inner_lower_target"] = (1, slice(None,None))
        
        
        # First SOL rings
        for name in ["outer", "outer_lower", "outer_upper", "inner", "inner_lower", "inner_upper"]:
            s[name] = self.make_custom_sol_ring(name, i = 0)
            
        # BOUT++ style regions - needs implementing leftcut, rightcut
        # s["lower_inner_pfr"] = (slice(None, leftcut[0]+2), slice(None, sep+2))
        # s["lower_inner_SOL"] = (slice(None, leftcut[0]+2), slice(sep+2, None))
        # s["inner_core"] = (slice(leftcut[0]+2, leftcut[1]+2), slice(None, sep+2))
        # s["inner_SOL"] = (slice(leftcut[0]+2, leftcut[1]+2), slice(sep+2, None))
        # s["upper_inner_PFR"] = (slice(leftcut[1]+2, upper_break), slice(None, sep+2))
        # s["upper_inner_SOL"] = (slice(leftcut[1]+2, upper_break), slice(sep+2, None))
        # s["upper_outer_PFR"] = (slice(upper_break, rightcut[1]+2), slice(None, sep+2))
        # s["upper_outer_SOL"] = (slice(upper_break, rightcut[1]+2), slice(sep+2, None))
        # s["outer_core"] = (slice(rightcut[1]+2, rightcut[0]+2), slice(None, sep+2))
        # s["outer_SOL"] = (slice(rightcut[1]+2, rightcut[0]+2), slice(sep+2, None))
        # s["lower_outer_PFR"] = (slice(rightcut[0]+2, None), slice(None, sep+2))
        # s["lower_outer_SOL"] = (slice(rightcut[0]+2, None), slice(sep+2, None))
            
        # ## Csalculate array of radial distance from separatrix
        # dist = np.cumsum(self.g["hy"][self.s["omp"]])   # hy is radial length
        # dist = dist - dist[self.g["sep"] - 1]
        # self.g["radial_dist"] = dist
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
        
        dfimp = species[species["name"].str.contains(f"{species_name}\+")]
        species_indices = list(dfimp.index)
        
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
        
        if all_states:
            print("Saving all states")
            dfimp = species[species["name"].str.contains(f"{species_name}")]
            for i in dfimp.index:
                species = dfimp.loc[i, "name"]
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
        bal["NVd+"] = bal["fnax_D+1_tot"] * Mi * 2 / bal["apar"]  # Parallel momentum flux kg/m2/s
        bal["Vd+"] = bal["NVd+"] / bal["Ne"] / (Mi*2)   # NOTE: this will break for multiple ions
        bal["M"] = bal["Vd+"] / np.sqrt((bal["Te"]*qe + bal["Td+"]*qe)/Mi*2)  # Mach number
        
                
        self.bal = bal
        self.params = list(self.bal.keys())


        
    def make_custom_sol_ring(self, name, i = None, sep_dist = None):
        """
        Inputs
        ------
            name, str:
                "outer", "outer_lower", "outer_upper", "inner", "inner_lower", "inner_upper"
            i, int: 
                SOL ring index (0 = first inside SOL) 
            sep_dist, int
                Separatrix distance in [m]
                
        Returns:
        ------
            selector tuple as (X,Y)
            
        NOTE: INCLUDES GUARD CELLS
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
        selections["inner"] =       (slice(1, self.g["upper_break"]), yid)
        selections["inner_lower"] = (slice(1, self.g["imp"]+1), yid)
        selections["inner_upper"] = (slice(self.g["imp"]-1, self.g["upper_break"]), yid)
        
        if name in selections:
            return selections[name]
        else:
            raise Exception(f"Region {name} not supported")
    

        
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
        
        if custom_cmap != None:
            cmap = custom_cmap
        
        if grid_only is True:
            data = np.zeros_like(self.bal["te"])
        elif len(data)==0:
            data = self.bal[param]
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
        
        
        # print(f"Data shape: {data.shape}")
        # print(f"Grid shape: {nx, ny}")
        # print(cry.shape)
        
        
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
        verbose = False
    ):
        """
        Returns OMP, IMP or target data from the balance file
        Note that the balance file shape is a transpose of the geometry shape (e.g. crx)
        and contains guard cells.        
        """

        if type(params) == str:
            params = [params]
        
        if any([region in name for name in ["omp", "imp", "outer_lower_target", "inner_lower_target"]]):
            p = self.s[region] 
        else:
            raise Exception(f"Unrecognised region: {region}")
        
        selector = (p[0], slice(None))
        df = pd.DataFrame()
        hy = self.g["hy"][selector] 
        
        # Interpolate between cells to get separatrix distance
        
        
        df = pd.DataFrame()
        
        for i, _ in enumerate(hy):
            if i == 0:
                df.loc[i, "dist"] = hy[i] / 2
            else:
                df.loc[i, "dist"] = df.loc[i-1, "dist"] + hy[i-1] + hy[i]/2
        
        sepind = self.g["sep"]
        sep_corr = (df["dist"][sepind] - df["dist"][sepind-1]) / 2
        dist_sep = df["dist"][sepind]
        df["dist"] -= dist_sep - sep_corr
        df["sep"] = 0
        df.loc[sepind, "sep"] = 1
        
        df["R"] = self.g["R"][selector]
        df["Z"] = self.g["Z"][selector]
        
        hx = self.bal["hx"][selector]
        Btot = self.g["Btot"][selector]
        Bpol = self.g["Bpol"][selector]
        R = self.g["R"][selector]
        vol = self.g["vol"][selector]
        
        dSpar = hx * abs(Btot / Bpol)
        df["apar"] = vol / dSpar

        for param in params:
            df[param] = self.bal[param][selector[0], :]
        
        return df
    
    def get_1d_poloidal_data(
        self,
        params,
        region = "outer_lower",
        sepadd = None,
        sepdist = None,
        target_first = False,
        
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
        
        ## REPLACED BY MAKE_CUSTOM_SOL_RING
        # if region == "outer_lower":
        #     selector = (slice(omp, -1), yind)
        # elif region == "inner_lower":
        #     selector = (slice(1, imp+1), yind)
        # elif region == "outer_upper":
        #     selector = (slice(1, imp+1), yind)
        # else:
        #     raise Exception("Unrecognised region")
        
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

        ## X-point
        if "inner" in region:
            idxmin = np.argmin(abs(R - R.max()))
        elif "outer" in region:
            idxmin = np.argmin(abs(R - R.min()))
        else:
            raise ValueError("Region not recognised")

        df["Xpoint"] = 0
        df.loc[idxmin, "Xpoint"] = 1

        if "inner" in region:
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

        if target_first:
            df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
            df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
            df = df.iloc[::-1]
        
        return df
        # return df[::-1]
    
    def find_param(self, name):
        """
        Returns variables that match string
        """
        for param in self.params:
            if name in param: print(param)
            
    def find_xpoints(self, plot = False):
        """
        Returns poloidal IDs of xpoints, works for double null only
        the IDs are for the cell immediately after the X-point
        in the direction of ascending X (poloidal) coordinate.
        They are arranged clockwise starting at inner lower
        
        NOT VERY ROBUST
        """

        x = range(self.g["nx"])
        R = self.g["R"][:,self.g["sep"]]

        inner_R = self.g["R"][slice(None, self.g["upper_break"]),self.g["sep"]]
        inner_x = np.array(range(self.g["upper_break"]))


        outer_R = self.g["R"][slice(self.g["upper_break"], None),self.g["sep"]]
        outer_x = np.array(range(self.g["upper_break"], self.g["nx"]))

        import scipy

        inner_ids = scipy.signal.find_peaks(inner_R, threshold = 0.0007)
        outer_ids = scipy.signal.find_peaks(outer_R*-1, threshold = 0.001)

            
                
        issue = False
        
        if (len(inner_ids[0]) > 2):
            print("Too many X-points found in inner SOL")
            issue = True
        elif (len(inner_ids[0]) < 2):
            print("Too few X-points found in inner SOL")
            issue = True
            
        if (len(outer_ids[0]) > 2):
            print("Too many X-points found in outer SOL")
            issue = True
            
        elif (len(outer_ids[0]) < 2):
            print("Too few X-points found in outer SOL")
            issue = True
            
        if issue or plot:
            fig, axes = plt.subplots(1,2, figsize = (8,4))
            ax = axes[0]
            ax.set_title("Inner SOL")
            ax.plot(inner_x, inner_R)
            for i in inner_ids[0]:
                ax.scatter(inner_x[i], inner_R[i])
            ax = axes[1]
            ax.set_title("Outer SOL")
            ax.plot(outer_x, outer_R)
            for i in outer_ids[0]:
                ax.scatter(outer_x[i], outer_R[i])
                
        if issue:
            raise Exception("Issue in peak finder, try reducing threshold")

        xpoints = [
            inner_ids[0][0], 
            inner_ids[0][1]+1, 
            outer_ids[0][0]+ self.g["upper_break"] + 1, 
            outer_ids[0][1]+ self.g["upper_break"]
            ]

        return xpoints
    
    def plot_selection(self, sel, **kwargs):
        """
        Plots scatter of selected points according to the tuple sel
        which contains slices in (X, Y)
        Examples provided in self.s
        """
        
        data = self.bal["ne"][:]
        data = np.zeros_like(data)

        data[sel] = 1
        
        def_kwargs = dict(
            separatrix_kwargs = dict(lw = 1, c = "darkorange"),
            linewidth = 0.1)
        
        plot_kwargs = {**def_kwargs, **kwargs}
        
        if "dpi" in kwargs:
            dpi = kwargs["dpi"]
        else:
            dpi = 150
        fig, ax = plt.subplots(dpi = dpi)

        self.plot_2d("", ax = ax, data = data, cmap = mpl.colors.ListedColormap(["lightgrey", "deeppink"]),
             antialias = True, 
             cbar = False,
             **plot_kwargs)
        
        ax.scatter(self.g["R"][sel], self.g["Z"][sel], c = "deeppink", s = 3)
        
        
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
    df[df["param"].str.contains("E\+")] = df[df["param"].str.contains("E\+")].shift(axis=1)

    params = df["param"][df["param"].str.contains("D0|D\+|None") == False].values

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

