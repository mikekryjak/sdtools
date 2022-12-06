#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from boututils.datafile import DataFile
from boututils.boutarray import BoutArray


class Mesh():
    # TODO identify guard cells in plots
    # TODO add separatrices to plots
    
    def __init__(self, filepath):
        """
        Initialise by reading mesh file
        """
        self.mesh = DataFile(filepath, write = True)
        self.fields = dict()    # For storing new fields to write
        self.filepath = filepath

        self.Rxy = self.mesh["Rxy"]    # R coordinate array
        self.Zxy = self.mesh["Zxy"]    # Z coordinate array
        
        self.ixseps1 = self.mesh["ixseps1"]
        self.MYG = self.mesh["y_boundary_guards"]
        self.MXG = 2
        self.ny_inner = self.mesh["ny_inner"]
        self.ny = self.mesh["ny"]
        self.nyg = self.ny + self.MYG * 4 # with guard cells
        self.nx = self.mesh["nx"]

        self.j1_1 = self.mesh["jyseps1_1"]
        self.j1_2 = self.mesh["jyseps1_2"]
        self.j2_1 = self.mesh["jyseps2_1"]
        self.j2_2 = self.mesh["jyseps2_2"]

        self.j1_1g = self.j1_1 + self.MYG
        self.j1_2g = self.j1_2 + self.MYG * 3
        self.j2_1g = self.j2_1 + self.MYG
        self.j2_2g = self.j2_2 + self.MYG * 3
        
        self.dx = self.mesh["dx"]
        self.dy = self.mesh["dy"]
        self.dydx = self.mesh["dy"] * self.mesh["dx"]    # Poloidal surface area
        self.J = self.mesh["J"]
        dz = 2*np.pi    # Axisymmetric
        self.dv = self.dydx * dz * self.mesh["J"]    # Cell volume

        # Array of radial (x) indices and of poloidal (y) indices in the style of Rxy, Zxy
        self.x_idx = np.array([np.array(range(self.nx))] * int(self.nyg)).transpose()
        self.y_idx = np.array([np.array(range(self.nyg))] * int(self.nx))

        self.yflat = self.y_idx.flatten()
        self.xflat = self.x_idx.flatten()
        self.rflat = self.Rxy.flatten()
        self.zflat = self.Zxy.flatten()
        

        self.colors = ["cyan", "lime", "crimson", "magenta"]

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


    def plot_slice(self, slice):
        """
        Indicates region of cells in X, Y and R, Z space for implementing Hermes-3 sources
        You must provide a slice() object for the X and Y dimensions which is a tuple in the form (X,Y)
        X is the radial coordinate (excl guards) and Y the poloidal coordinate (incl guards)
        WARNING: only developed for a connected double null. Someone can adapt this to a single null or DDN.
        """

        mesh = self.mesh 
        xslice = slice[0]
        yslice = slice[1]

        # Region boundaries
        ny = mesh["ny"]     # Total ny cells (incl guard cells)
        nx = mesh["nx"]     # Total nx cells (excl guard cells)
        Rxy = mesh["Rxy"]    # R coordinate array
        Zxy = mesh["Zxy"]    # Z coordinate array
        MYG = mesh["y_boundary_guards"]

        # Array of radial (x) indices and of poloidal (y) indices in the style of Rxy, Zxy
        x_idx = np.array([np.array(range(nx))] * int(ny + MYG * 4)).transpose()
        y_idx = np.array([np.array(range(ny + MYG*4))] * int(nx))

        # Slice the X, Y and R, Z arrays and vectorise them for plotting
        xselect = x_idx[xslice,yslice].flatten()
        yselect = y_idx[xslice,yslice].flatten()
        rselect = Rxy[xslice,yslice].flatten()
        zselect = Zxy[xslice,yslice].flatten()

        # Plot
        fig, axes = plt.subplots(1,3, figsize=(12,5), dpi = 120, gridspec_kw={'width_ratios': [2.5, 1, 2]})
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
        ax.legend(loc = "upper center", bbox_to_anchor = (0.5,-0.1), ncol = 4)
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

        if xlim != (None,None):
            ax.set_xlim(xlim)
        if ylim != (None,None):
            ax.set_ylim(ylim)

    def write_field(self, field):

        data = field.data
        data = np.expand_dims(data, -1)    # Add Z dimension of 1
        data.attributes["bout_type"] = "Field3D"
        self.mesh.write(field.name, data, info = True)

        if field.name in self.mesh.list():
            print(f">>> Field {field.name} already exists in {self.filepath}, it will be overwritten")
        print(f"-> Wrote field {field.name} to {self.filepath}")
        
    def summarise_grid(self):
        meta = self.mesh
        print(f' - ixseps1: {meta["ixseps1"]}    // id of first cell after separatrix 1')
        print(f' - ixseps2: {meta["ixseps2"]}    // id of first cell after separatrix 2')
        print(f' - jyseps1_1: {meta["jyseps1_1"]}    // near lower inner')
        print(f' - jyseps1_2: {meta["jyseps1_2"]}    // near lower outer')
        print(f' - jyseps2_1: {meta["jyseps2_1"]}    // near upper outer')
        print(f' - jyseps2_2: {meta["jyseps2_2"]}    // near lower outer')
        print(f' - ny_inner: {meta["ny_inner"]}    // no. poloidal cells in-between divertor regions')
        print(f' - ny: {meta["ny"]}    // total cells in Y (poloidal, does not include guard cells)')
        print(f' - nx: {meta["nx"]}    // total cells in X (radial, includes guard cells)')


class Field():
    def __init__(self, name, mesh):
        self.name = name
        self.mesh = mesh
        self.data = np.zeros_like(self.mesh.Rxy)    # Copy any array from existing grid as a template


    def plot(self):

        plt.style.use("default")


        field = self.data
        cmap = plt.get_cmap("jet")


        fieldnorm = field / np.max(field) * 1
        colors = [cmap(x) for x in fieldnorm.flatten()]
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(field))

        fig, axes = plt.subplots(1,3, figsize = (10,6), gridspec_kw={'width_ratios': [5,2.0, 0.3]}, dpi = 110)
        fig.subplots_adjust(wspace=0.3)
        fig.suptitle(self.name)

        self.mesh.plot_xy_grid(axes[0])
        axes[0].scatter(self.mesh.yflat, self.mesh.xflat, c = colors, s = 1)

        self.mesh.plot_rz_grid(axes[1])
        axes[1].scatter(self.mesh.rflat, self.mesh.zflat, c = colors, s = 5)

        cbar = mpl.colorbar.ColorbarBase(ax=axes[2], cmap = cmap, norm = norm)

        




