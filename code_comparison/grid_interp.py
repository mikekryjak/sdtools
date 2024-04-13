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
from code_comparison.solps_pp import *

from gridtools.solps_python_scripts.read_b2fgmtry import *

# import gridtools.solps_python_scripts.setup
# from gridtools.solps_python_scripts.plot_solps       import plot_1d, plot_2d, plot_wall_loads
# from gridtools.solps_python_scripts.read_ft44 import read_ft44
import ipywidgets as widgets

# from solps_python_scripts.read_b2fgmtry import read_b2fgmtry


class TransplantFromSOLPS:

    def __init__(self, hermes_ds, solps_path):
        self.hermes_ds = hermes_ds
        self.solps_case = SOLPScase(solps_path)
        self.solps_g = read_b2fgmtry(
            where=solps_path, save=False, ghost_cells=True, force=True
        )

        self.get_hermes_regions()

    def align_psi(self, diagnose=False):
        """
        Correct COCOS in SOLPS psi so that it matches Hermes-3

        - Hermes-3 has revesed Bpol and Btor which means the psi gradient is reversed
        - There is also a 2pi factor in SOLPS due to different cocos
        - There is also a different offset
        - The above shouldn't matter, so below we align the two
        """

        ds = self.hermes_ds
        spc = self.solps_case
        g = self.solps_g

        ### Hermes-3
        omp = ds.hermesm.select_region("outer_midplane_a")
        hsep = ds.metadata["ixseps1"]
        hpsi = omp["psi_poloidal"].values
        # hdist = np.cumsum(omp["dr"]).values
        hdist = omp["R"].values
        hdist -= hdist[hsep]

        ### SOLPS-ITER
        ## Get psi and normalise it to separatrix
        spsi = g["fpsi"].mean(axis=2).squeeze()
        spsi /= 2 * np.pi

        ## Reverse and offset to match Hermes-3
        #  Offset is based on separatrix midplane position
        spsi *= -1
        p = spc.s["omp"]
        spsiomp = spsi[p[0], :]
        ssep = int(g["topcut"][0])
        spsi -= spsiomp[ssep] - hpsi[hsep]
        spsiomp = spsi[p[0], :]
        sdist = spc.g["R"][p[0], p[1]] - spc.g["R"][p[0], spc.g["sep"]]

        # Plot
        if diagnose is True:
            fig, ax = plt.subplots()
            ax.plot(hdist, hpsi, label="Hermes-3")
            ax.plot(sdist, spsiomp, label="SOLPS-ITER")
            ax.legend()
            ax.set_xlabel("R")
            ax.set_ylabel("psi")
            ax.vlines(
                0,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                linestyle="--",
                color="darkslategrey",
            )
            ax.set_title("Psi at OMP")

        self.psi_solps = spsi
        print("Psi aligned")

    def get_hermes_regions(self):

        ds = self.hermes_ds
        m = ds.metadata

        sep = m["ixseps1"]
        hregion_sel = {}
        hregion_sel["lower_inner_SOL"] = (
            slice(sep, -m["MXG"]),
            slice(m["MYG"], m["j1_1g"] + 1),
        )
        hregion_sel["lower_inner_PFR"] = (
            slice(m["MXG"], sep),
            slice(m["MYG"], m["j1_1g"] + 1),
        )
        hregion_sel["inner_core"] = (
            slice(m["MXG"], sep),
            slice(m["j1_1g"] + 1, m["j2_1g"] + 1),
        )
        hregion_sel["inner_SOL"] = (
            slice(sep, -m["MXG"]),
            slice(m["j1_1g"] + 1, m["j2_1g"] + 1),
        )
        hregion_sel["upper_inner_PFR"] = (
            slice(m["MXG"], sep),
            slice(m["j2_1g"] + 1, m["ny_inner"]),
        )
        hregion_sel["upper_inner_SOL"] = (
            slice(sep, -m["MXG"]),
            slice(m["j2_1g"] + 1, m["ny_inner"]),
        )
        hregion_sel["upper_outer_PFR"] = (
            slice(m["MXG"], sep),
            slice(m["ny_inner"] + m["MYG"] * 3, m["j1_2g"] + 1),
        )
        hregion_sel["upper_outer_SOL"] = (
            slice(sep, -m["MXG"]),
            slice(m["ny_inner"] + m["MYG"] * 3, m["j1_2g"] + 1),
        )
        hregion_sel["outer_core"] = (
            slice(m["MXG"], sep),
            slice(m["j1_2g"] + 1, m["j2_2g"] + 1),
        )
        hregion_sel["outer_SOL"] = (
            slice(sep, -m["MXG"]),
            slice(m["j1_2g"] + 1, m["j2_2g"] + 1),
        )
        hregion_sel["lower_outer_PFR"] = (
            slice(m["MXG"], sep),
            slice(m["j2_2g"] + 1, -m["MYG"]),
        )
        hregion_sel["lower_outer_SOL"] = (
            slice(sep, -m["MXG"]),
            slice(m["j2_2g"] + 1, -m["MYG"]),
        )

        hgroup_sel = {}
        hgroup_sel["lower_inner_PFR"] = hregion_sel["lower_inner_PFR"]
        hgroup_sel["inner_full_SOL"] = (
            slice(sep, -m["MXG"]),
            slice(m["MYG"], m["ny_inner"]),
        )
        hgroup_sel["inner_core"] = hregion_sel["inner_core"]
        hgroup_sel["upper_inner_PFR"] = hregion_sel["upper_inner_PFR"]
        hgroup_sel["upper_outer_PFR"] = hregion_sel["upper_outer_PFR"]
        hgroup_sel["outer_full_SOL"] = (
            slice(sep, -m["MXG"]),
            slice(m["ny_inner"] + m["MYG"] * 3, -m["MYG"]),
        )
        hgroup_sel["outer_core"] = hregion_sel["outer_core"]
        hgroup_sel["lower_outer_PFR"] = hregion_sel["lower_outer_PFR"]

        self.hregion_sel = hregion_sel
        self.hgroup_sel = hgroup_sel
        
    def get_flux_coordinates(self):
        ds = self.hermes_ds
        m = ds.metadata
        
        ## Set up X and Y
        hx = ds["psi_poloidal"].values
        hy = ds["theta_idx"].values
        hy = np.expand_dims(hy, axis=0)
        hy = np.expand_dims(hy, axis=2)
        hy = np.repeat(hy, hx.shape[0], axis=0)
        hx = hx.squeeze()
        hy = hy.squeeze()
        

    def plot_hermes_regions_poloidal(self, ax, regions):

        ds = self.hermes_ds
        m = ds.metadata
        
        ## Set up X and Y
        x = ds["psi_poloidal"].values
        y = ds["theta_idx"].values
        y = np.expand_dims(y, axis=0)
        y = np.expand_dims(y, axis=2)
        y = np.repeat(y, x.shape[0], axis=0)
        x = x.squeeze()
        y = y.squeeze()

        ## Set up data, zeros based on Ne and coloured by region
        data = ds["Ne"].copy()
        data.loc[:] = 0

        for i, key in enumerate(regions):
            i += 1
            data.data[regions[key]] = i
            
        ## Colormap
        colors = list(mpl.colormaps["turbo"](np.linspace(0, 1, len(regions.keys()))))
        cmap = mpl.colors.ListedColormap(["white"] + colors)

        data.bout.polygon(
            ax=ax,
            separatrix=False,
            targets=False,
            cmap=cmap,
            antialias=True,
            add_colorbar=False,
            linewidth=0.1,
            vmin=0,
            vmax=len(regions.keys()),
        )
        
        ax.set_xlim(0.12, 0.82)
        ax.set_ylim(-0.86, 0.86)
        
    def plot_hermes_regions_flux(self, ax, regions):
        
        ## Colormap
        colors = list(mpl.colormaps["turbo"](np.linspace(0, 1, len(regions.keys()))))
        cmap = mpl.colors.ListedColormap(["white"] + colors)
        
        for i, region in enumerate(regions):

            slices = regions[region]
            x_region = x[slices]
            y_region = y[slices]

            ax.pcolormesh(
                x_region,
                y_region,
                np.ones_like(x_region),
                cmap=mpl.colors.ListedColormap(colors[i]),
                vmin=0,
                vmax=len(regions.keys()),
                rasterized=True,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
                shading="nearest",
            )
        
