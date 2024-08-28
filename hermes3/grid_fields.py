#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from boututils.datafile import DataFile
from boututils.boutarray import BoutArray
import shutil
from hermes3.utils import *


def make_new_mesh(source, destination):
    # Create new grid from an existing one, read it in and create the Field object
    shutil.copy(source, destination)
    return Mesh(destination)
    
def close_mesh(mesh):
    try:
        mesh.close()
        del mesh
    except:
        pass
    
def impose_fields(source, destination,
                  Nd_src_puff = 0,   # s-1
                  Pd_src_puff = 0,   # Pa of pressure
                  Ni_src_core = 0,   # s-1 of particles
                  Pi_src_core = 0,   # Pa of pressure
                  Pe_src_core = 0,   # Pa of pressure
                  D_core = 0,    # m2s-1
                  D_sol  = 0,    # ms2s-1
                  chi_core = 0,   # m2s-1
                  chi_sol = 0,    # m2s-1
                  
                  
                  
                  ):
    """
    Copy a mesh from source to destination and impose fields
    Hardcoded for now
    """
    
    # close_mesh()
    
    mesh = make_new_mesh(source, destination)
    
    # Outboard D puff
    # Nd_src_puff = 1.2e21    # s-1
    # Pd_src_puff = 2/3 * 1.2e21 * 3 * constants("q_e")  #  3eV per particle (assume dissociated molecules). Remember this is pressure not energy

    # Core particle/heat sources
    # Ni_src_core = 3e20    # s-1
    # Pi_src_core = 1e6 * 2/3   # W converted to pressure
    # Pe_src_core = 0.76e6 * 2/3   # W converted to pressure

    # Anomalous diffusion coefficients
    # D_core = 0.3
    # chi_core = 0.45
    # D_sol = 1
    # chi_sol = 3
    print("WARNING: Puff is suppressed")
    # Make regions
    # puff_region = mesh.slices("symmetric_puff")(width=3, center_half_gap=1)
    core_edge_region = mesh.slices("core_edge")
    core_region = mesh.slices("core")
    sol_region = mesh.slices("sol")
    pfr_region = mesh.slices("pfr")
    fields = dict()
    
    # fields["Nd_src"] = Field("Nd_src", mesh)
    # fields["Nd_src"].set_value(puff_region, Nd_src_puff, make_per_volume = True)

    # fields["Pd_src"] = Field("Pd_src", mesh)
    # fields["Pd_src"].set_value(puff_region, Pd_src_puff, make_per_volume = True)

    fields["Nd+_src"] = Field("Nd+_src", mesh)
    fields["Nd+_src"].set_value(core_edge_region, Ni_src_core, make_per_volume = True)

    fields["Pd+_src"] = Field("Pd+_src", mesh)
    fields["Pd+_src"].set_value(core_edge_region, Pi_src_core, make_per_volume = True)

    fields["Pe_src"] = Field("Pe_src", mesh)
    fields["Pe_src"].set_value(core_edge_region, Pe_src_core, make_per_volume = True)

    fields["D_d+"] = Field("D_d+", mesh)
    fields["D_d+"].set_value(core_region, D_core, make_per_volume = False)
    fields["D_d+"].set_value(sol_region, D_sol, make_per_volume = False)
    fields["D_d+"].set_value(pfr_region, D_sol, make_per_volume = False)

    fields["D_e"] = Field("D_e", mesh)
    fields["D_e"].set_value(core_region, D_core, make_per_volume = False)
    fields["D_e"].set_value(sol_region, D_sol, make_per_volume = False)
    fields["D_e"].set_value(pfr_region, D_sol, make_per_volume = False)

    fields["chi_d+"] = Field("chi_d+", mesh)
    fields["chi_d+"].set_value(core_region, chi_core, make_per_volume = False)
    fields["chi_d+"].set_value(sol_region, chi_sol, make_per_volume = False)
    fields["chi_d+"].set_value(pfr_region, chi_sol, make_per_volume = False)

    fields["chi_e"] = Field("chi_e", mesh)
    fields["chi_e"].set_value(core_region, chi_core, make_per_volume = False)
    fields["chi_e"].set_value(sol_region, chi_sol, make_per_volume = False)
    fields["chi_e"].set_value(pfr_region, chi_sol, make_per_volume = False)
    
    for field_name in fields.keys():
        
        if "D_" in field_name or "chi_" in field_name:
            mesh.write_field(fields[field_name], dtype = "Field2D")
        else:
            mesh.write_field(fields[field_name], dtype = "Field3D")
        
        fields[field_name].plot()
        
    close_mesh(mesh)
        

class Mesh():
    """ 
    Wrapper for the Mesh DataFile
    Opens a mesh .nc datafile and performs operations on it, such as writing
    custom fields and plotting them.
    NOTE after you're done with operations, you must manually close the datafile
    using Mesh.close(). This prevents unexpected behaviour such as the inability
    to reload mesh files after changes.
    """
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
        self.ixseps2 = self.mesh["ixseps2"]
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
        

        self.colors = ["cyan", "lime", "crimson", "magenta", "black", "red"]
        
    def close(self):
        """
        Close the opened mesh datafile.
        """
        self.mesh.close()
        print(f"Mesh file {self.filepath} closed")

    def slices(self, name):
        """
        DOUBLE NULL ONLY
        Pass this touple to a field of any parameter spanning the grid
        to select points of the appropriate region.
        Each slice is a tuple: (x slice, y slice)
        Use it as: selected_array = array[slice] where slice = (x selection, y selection) = output from this method.
        """
        
        def upper_seam(i):
            """
            Creates a slice for the upper mesh seam
            
            Inputs
            ------
            i: int
                Extent of selected region on each side of the seam in cell number. 
                Total number of cells selected will be 2i.
            """
            
            MXG = self.MXG
            j1_2g = self.j1_2g
            j2_1g = self.j2_1g
            
            
            return (slice(0+MXG,1+MXG), 
                    np.r_[
                            slice(j1_2g + 1, j1_2g + (i+1)) , 
                            slice(j2_1g + 1 - (i), j2_1g + 1)
                            ])
            
            
        def lower_seam(i):
            """
            Creates a slice for the lower mesh seam
            
            Inputs
            ------
            i: int
                Extent of selected region on each side of the seam in cell number. 
                Total number of cells selected will be 2i.
            """
            
            MXG = self.MXG
            j1_1g = self.j1_1g
            j2_2g = self.j2_2g
            
            
            return (slice(0+MXG,1+MXG), 
                    np.r_[
                            slice(j2_2g + 1 - i, j2_2g + 1) , 
                            slice(j1_1g + 1, j1_1g + 1 + i)
                            ])
            
            
        
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
        
        def symmetric_puff(width, center_half_gap):
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
            return selection
            # return self.ds.isel(x = selection[0], theta = selection[1])
                

        slices = dict()
        
        slices["custom_core_ring"] = custom_core_ring
        slices["symmetric_puff"] = symmetric_puff
        slices["upper_seam"] = upper_seam
        slices["lower_seam"] = lower_seam

        slices["all"] = (slice(None,None), slice(None,None))
        slices["all_noguards"] = (slice(self.MXG,-self.MXG), np.r_[slice(self.MYG,self.ny_inner-self.MYG*2), slice(self.ny_inner+self.MYG*3, self.nyg - self.MYG)])

        slices["inner_core"] = (slice(0,self.ixseps1), np.r_[slice(self.j1_1g + 1, self.j2_1g+1), slice(self.j1_2g + 1, self.j2_2g + 1)])
        slices["outer_core"] = (slice(self.ixseps1, None), slice(0, self.nyg))
        
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

        return slices[name]


    def plot_slice(self, slice, dpi = 100):
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
        ny = self.ny      # Total ny cells (incl guard cells)
        nx = self.nx    # Total nx cells (excl guard cells)
        Rxy = self.Rxy    # R coordinate array
        Zxy = self.Zxy    # Z coordinate array
        MYG = self.MYG

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
        ax.legend(loc = "upper center", bbox_to_anchor = (0.5,-0.1), ncol = 4)
        ax.set_xlabel("Y index (incl. guards)")
        ax.set_ylabel("X index (excl. guards)")

    # def plot_rz_grid(self, ax, xlim = (None,None), ylim = (None,None)):
    #     ax.set_title("R, Z space")
    #     ax.scatter(self.rflat, self.zflat, s = 0.1, c = "black")
    #     ax.set_axisbelow(True)
    #     ax.grid()
    #     ax.plot(self.Rxy[:,self.j1_1g], self.Zxy[:,self.j1_1g], label = "j1_1g",     color = self.colors[0], alpha = 0.7)
    #     ax.plot(self.Rxy[:,self.j1_2g], self.Zxy[:,self.j1_2g], label = "j1_2g", color = self.colors[1], alpha = 0.7)
    #     ax.plot(self.Rxy[:,self.j2_1g], self.Zxy[:,self.j2_1g], label = "j2_1g",     color = self.colors[2], alpha = 0.7)
    #     ax.plot(self.Rxy[:,self.j2_2g], self.Zxy[:,self.j2_2g], label = "j2_2g", color = self.colors[3], alpha = 0.7)
    #     ax.plot(self.Rxy[self.ixseps1,:], self.Zxy[self.ixseps1,:], label = "ixseps1", color = self.colors[4], alpha = 0.7, lw = 2)
    #     ax.plot(self.Rxy[self.ixseps2,:], self.Zxy[self.ixseps2,:], label = "ixseps2", color = self.colors[5], alpha = 0.7, lw = 2, ls=":")

    #     if xlim != (None,None):
    #         ax.set_xlim(xlim)
    #     if ylim != (None,None):
    #         ax.set_ylim(ylim)
            
    def plot_rz_grid(self, ax, 
                     xlim = (None,None), ylim = (None,None),
                     linecolor = "k",
                     plot_region_borders = True):
        
        linewidth = 0.5
        ax.set_title("R, Z space")
        
        
        if "Rxy_lower_right_corners" in self.mesh.keys():
            r_nodes = [
                "Rxy",
                "Rxy_corners",
                "Rxy_lower_right_corners",
                "Rxy_upper_left_corners",
                "Rxy_upper_right_corners",
            ]
            z_nodes = [
                "Zxy",
                "Zxy_corners",
                "Zxy_lower_right_corners",
                "Zxy_upper_left_corners",
                "Zxy_upper_right_corners",
            ]
            cell_r = np.concatenate(
                [np.expand_dims(self.mesh[x], axis=2) for x in r_nodes], axis=2
            )
            cell_z = np.concatenate(
                [np.expand_dims(self.mesh[x], axis=2) for x in z_nodes], axis=2
            )
        else:
            raise Exception("Cell corners not present in mesh, cannot do polygon plot")

        Nx = len(cell_r)
        Ny = len(cell_r[0])
        patches = []

        # https://matplotlib.org/2.0.2/examples/api/patch_collection.html

        idx = [np.array([1, 2, 4, 3, 1])]
        patches = []
        for i in range(Nx):
            for j in range(Ny):
                p = mpl.patches.Polygon(
                    np.concatenate((cell_r[i][j][tuple(idx)], cell_z[i][j][tuple(idx)]))
                    .reshape(2, 5)
                    .T,
                    fill=False,
                    closed=True,
                    facecolor=None,
                )
                patches.append(p)
                
        cmap = mpl.colors.ListedColormap(["white"])
        colors =np.zeros_like(cell_r).flatten()
        polys = mpl.collections.PatchCollection(
            patches,
            alpha=0.5,
            # norm=norm,
            cmap=cmap,
            # fill = False,
            antialiaseds=True,
            edgecolors=linecolor,
            linewidths=linewidth,
            joinstyle="bevel",
        )

        polys.set_array(colors)
        ax.add_collection(polys)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_ylim(cell_z.min(), cell_z.max())
        ax.set_xlim(cell_r.min(), cell_r.max())
        
        ax.set_axisbelow(True)
        ax.grid()
        
        if plot_region_borders is True:
            ax.plot(self.Rxy[:,self.j1_1g], self.Zxy[:,self.j1_1g], label = "j1_1g",     color = self.colors[0], alpha = 0.7)
            ax.plot(self.Rxy[:,self.j1_2g], self.Zxy[:,self.j1_2g], label = "j1_2g", color = self.colors[1], alpha = 0.7)
            ax.plot(self.Rxy[:,self.j2_1g], self.Zxy[:,self.j2_1g], label = "j2_1g",     color = self.colors[2], alpha = 0.7)
            ax.plot(self.Rxy[:,self.j2_2g], self.Zxy[:,self.j2_2g], label = "j2_2g", color = self.colors[3], alpha = 0.7)
            ax.plot(self.Rxy[self.ixseps1,:], self.Zxy[self.ixseps1,:], label = "ixseps1", color = self.colors[4], alpha = 0.7, lw = 1)
            ax.plot(self.Rxy[self.ixseps2,:], self.Zxy[self.ixseps2,:], label = "ixseps2", color = self.colors[5], alpha = 0.7, lw = 1, ls=":")

        if xlim != (None,None):
            ax.set_xlim(xlim)
        if ylim != (None,None):
            ax.set_ylim(ylim)
            
    def plot_field(self, name):
        """
        Plot a field saved in the grid file
        """
        
        plt.style.use("default")

        field = self.mesh[name]
        cmap = plt.get_cmap("Spectral_r")

        fieldnorm = field / np.max(field) * 1
        colors = [cmap(x) for x in fieldnorm.flatten()]
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(field))

        fig, axes = plt.subplots(1,3, figsize = (10,6), gridspec_kw={'width_ratios': [5,2.0, 0.3]}, dpi = 110)
        fig.subplots_adjust(wspace=0.3)
        fig.suptitle(name)

        self.plot_xy_grid(axes[0])
        axes[0].scatter(self.yflat, self.xflat, c = colors, s = 1)

        self.plot_rz_grid(axes[1])
        axes[1].scatter(self.rflat, self.zflat, c = colors, s = 5)

        cbar = mpl.colorbar.ColorbarBase(ax=axes[2], cmap = cmap, norm = norm)

    def write_field(self, field, dtype = "Field3D"):

        data = field.data
        
        if dtype == "Field3D":
            data = np.expand_dims(data, -1)    # Add Z dimension of 1
            data.attributes["bout_type"] = "Field3D"
            
        elif dtype == "Field2D":
            data.attributes["bout_type"] = "Field2D"
            
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

    def set_value(self, region, value, make_per_volume = True):
        
        cell_volumes = self.mesh.dv[region]
        total_volume = cell_volumes.sum()
        
        if make_per_volume is True:
            value = value * (cell_volumes/total_volume)  # Split between all the cells in region according to their volumes
            value = value / cell_volumes   # Make on a per volume basis
            
        self.data[region] = value
        
        
        
        
    def plot(self, dpi = 80):

        plt.style.use("default")


        field = self.data
        cmap = plt.get_cmap("YlOrRd")


        fieldnorm = field / np.max(field) * 1
        colors = [cmap(x) for x in fieldnorm.flatten()]
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(field))

        fig, axes = plt.subplots(1,3, figsize = (10,6), gridspec_kw={'width_ratios': [5,2.0, 0.3]}, dpi = dpi)
        fig.subplots_adjust(wspace=0.3)
        fig.suptitle(self.name)

        self.mesh.plot_xy_grid(axes[0])
        axes[0].scatter(self.mesh.yflat, self.mesh.xflat, c = colors, s = 1)

        self.mesh.plot_rz_grid(axes[1])
        axes[1].scatter(self.mesh.rflat, self.mesh.zflat, c = colors, s = 4)

        cbar = mpl.colorbar.ColorbarBase(ax=axes[2], cmap = cmap, norm = norm)
        
        
def compare_grid(
    mesh1_path,
    mesh2_path,
    fields = ["Nd+_src", "Pd+_src", "Pe_src", "D_d+", "D_e", "chi_d+", "chi_e"]):
    mesh1 = Mesh(mesh1_path)
    mesh2 = Mesh(mesh2_path)

    for field in fields:
        print("\n")
        print(field, (mesh1.mesh[field] == mesh2.mesh[field]).all())

        print(np.mean(mesh1.mesh[field]))
        print(np.mean(mesh2.mesh[field]))
        
    mesh1.close()
    mesh2.close()

    try:
        del mesh1
    except:
        pass
    try:
        del mesh2
    except:
        pass


        




