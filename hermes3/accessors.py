from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
from hermes3.plotting import *
import numpy as np


@register_dataarray_accessor("hermesm")
class HermesDataArrayAccessor(BoutDataArrayAccessor):
    """
    Methods on Hermes-3 dataarrays
    """

    def __init__(self, da):
        super().__init__(da)

    def select_region(self, name):
        selection = _select_region(self.data, name)
        return self.data.isel(x=selection[0], theta=selection[1])

    def select_custom_core_ring(self, i):
        return _select_custom_core_ring(self.data, i)

    def clean_guards(self):
        """
        Set guard cell values to np.nan
        Implemented only for x guards for now
        """
        xguards = _select_region(self.data, "xguards")
        ds = self.data.copy()
        ds[{"x": xguards[0], "theta": xguards[1]}] = np.nan
        return ds

    def plot_omp(self):
        plot_omp([self.data], legend = False, title = False)
        

@register_dataset_accessor("hermesm")
class HermesDatasetAccessor(BoutDatasetAccessor):
    """
    Methods on Hermes-3 datasets
    """

    def __init__(self, ds):
        super().__init__(ds)

    def select_region(self, name):
        selection = _select_region(self.data, name)
        return self.data.isel(x=selection[0], theta=selection[1])

    def select_custom_core_ring(self, i):
        return _select_custom_core_ring(self.data, i)
    
    def select_custom_sol_ring(self, i, region):
        return _select_custom_sol_ring(self.data, i, region)


def _select_region(ds, name):
    """
    DOUBLE NULL ONLY
    Pass this tuple to a field of any parameter spanning the grid
    to select points of the appropriate region.
    Each slice is a tuple: (x slice, y slice)
    Use it as: selected_array = array[slice] where slice = (x selection, y selection) = output from this method.
    Returns sliced xarray dataset
    NOTE: This wasn't thoroughly tested for cases without guard cells or single nulls
    """
    m = ds.metadata

    if m["keep_xboundaries"] == 1:
        MXG = m["MXG"]
    else:
        MXG = 0

    if m["keep_yboundaries"] == 1:
        MYG = m["MYG"]
    else:
        MYG = 0

    ixseps1 = m["ixseps1"]
    ny_inner = m["ny_inner"]
    ny = m["ny"]
    nyg = ny + MYG * 4  # with guard cells
    nx = m["nx"]

    # Array of radial (x) indices and of poloidal (y) indices in the style of Rxy, Zxy
    x_idx = np.array([np.array(range(nx))] * int(nyg)).transpose()
    y_idx = np.array([np.array(range(nyg))] * int(nx))

    yflat = y_idx.flatten()
    xflat = x_idx.flatten()
    rflat = ds.coords["R"].values.flatten()
    zflat = ds.coords["Z"].values.flatten()

    j1_1 = m["jyseps1_1"]
    j1_2 = m["jyseps1_2"]
    j2_1 = m["jyseps2_1"]
    j2_2 = m["jyseps2_2"]
    ixseps2 = m["ixseps2"]
    ixseps1 = m["ixseps1"]
    Rxy = ds.coords["R"]
    Zxy = ds.coords["Z"]

    j1_1g = j1_1 + MYG
    j1_2g = j1_2 + MYG * 3
    j2_1g = j2_1 + MYG
    j2_2g = j2_2 + MYG * 3
    
    if all([x in name for x in ["target", "guard"]]) and MYG == 0:
        raise Exception("Trying to select yguards on a dataset without any")
    
    if all([x in name for x in ["xguards"]]) and MXG == 0:
        raise Exception("Trying to select xguards on a dataset without any")

    slices = dict()

    slices["all"] = (slice(None, None), slice(None, None))
    slices["all_noguards"] = (
        slice(MXG, -MXG),
        np.r_[slice(MYG, ny_inner - MYG * 2), slice(ny_inner + MYG * 3, nyg - MYG)],
    )

    slices["core"] = (
        slice(0, ixseps1),
        np.r_[slice(j1_1g + 1, j2_1g + 1), slice(j1_2g + 1, j2_2g + 1)],
    )
    slices["core_noguards"] = (
        slice(MXG, ixseps1),
        np.r_[slice(j1_1g + 1, j2_1g + 1), slice(j1_2g + 1, j2_2g + 1)],
    )
    slices["sol"] = (slice(ixseps1, None), slice(0, nyg))
    slices["sol_noguards"] = (
        slice(ixseps1, -MYG),
        np.r_[slice(MYG, ny_inner - MYG * 2), slice(ny_inner + MYG * 3, nyg - MYG)],
    )

    slices["outer_core_edge"] = (slice(0 + MXG, 1 + MXG), slice(j1_2g + 1, j2_2g + 1))
    slices["inner_core_edge"] = (slice(0 + MXG, 1 + MXG), slice(j1_1g + 1, j2_1g + 1))
    slices["core_edge"] = (
        slice(0 + MXG, 1 + MXG),
        np.r_[slice(j1_2g + 1, j2_2g + 1), slice(j1_1g + 1, j2_1g + 1)],
    )

    if MXG != 0:
        slices["outer_sol_edge"] = (
            slice(-1 - MXG, -MXG),
            slice(ny_inner + MYG * 3, nyg - MYG),
        )
        slices["inner_sol_edge"] = (slice(-1 - MXG, -MXG), slice(MYG, ny_inner + MYG))
        slices["sol_edge"] = (
            slice(-1 - MXG, -MXG),
            np.r_[slice(0, ny_inner), slice(ny_inner + MYG * 3, nyg - MYG)],
        )
        slices["xguards"] = (
            np.r_[slice(0, MXG), slice(nx - MXG, nx)],
            slice(None, None),
        )

    else:
        slices["outer_sol_edge"] = (
            slice(-1, None),
            slice(ny_inner + MYG * 3, nyg - MYG),
        )
        slices["inner_sol_edge"] = (slice(-1, None), slice(MYG, ny_inner + MYG))
        slices["sol_edge"] = (
            slice(-1 - MXG, -MXG),
            np.r_[slice(0, j2_1g + 1), slice(ny_inner + MYG * 3, nyg - MYG)],
        )
        slices["xguards"] = (0, 0)

    slices["inner_lower_target"] = (slice(None, None), slice(MYG, MYG + 1))
    slices["inner_upper_target"] = (
        slice(None, None),
        slice(ny_inner + MYG - 1, ny_inner + MYG),
    )
    slices["outer_upper_target"] = (
        slice(None, None),
        slice(ny_inner + MYG * 3, ny_inner + MYG * 3 + 1),
    )
    slices["outer_lower_target"] = (slice(None, None), slice(nyg - MYG - 1, nyg - MYG))

    slices["inner_lower_target_guard"] = (slice(None, None), slice(MYG - 1, MYG))
    slices["inner_upper_target_guard"] = (
        slice(None, None),
        slice(ny_inner + MYG, ny_inner + MYG + 1),
    )
    slices["outer_upper_target_guard"] = (
        slice(None, None),
        slice(ny_inner + MYG * 3 - 1, ny_inner + MYG * 3),
    )
    slices["outer_lower_target_guard"] = (
        slice(None, None),
        slice(nyg - MYG, nyg - MYG + 1),
    )

    slices["inner_lower_pfr"] = (slice(0, ixseps1), slice(None, j1_1g))
    slices["outer_lower_pfr"] = (slice(0, ixseps1), slice(j2_2g + 1, nyg))

    slices["lower_pfr"] = (
        slice(0, ixseps1),
        np.r_[slice(None, j1_1g + 1), slice(j2_2g + 1, nyg)],
    )
    slices["inner_lower_pfr_edge"] = (
        slice(MXG, MXG + 1),
        slice(None, j1_1g + 1),
    )
    slices["outer_lower_pfr_edge"] = (
        slice(MXG, MXG + 1),
        slice(j2_2g + 1, nyg),
    )

    slices["upper_pfr"] = (slice(0, ixseps1), slice(j2_1g + 1, j1_2g + 1))

    if "single-null" not in m["topology"]:
        slices["inner_upper_pfr_edge"] = (
            slice(MXG, MXG + 1),
            slice(j2_1g + 1, ny_inner),
        )
        
        slices["outer_upper_pfr_edge"] = (
            slice(MXG, MXG + 1),
            slice(ny_inner + MYG * 2, j1_2g + 1),
        )


    slices["pfr"] = (
        slice(0, ixseps1),
        np.r_[
            np.r_[slice(None, j1_1g + 1), slice(j2_2g + 1, nyg)],
            slice(j2_1g + 1, j1_2g + 1),
        ],
    )

    slices["lower_pfr_edge"] = (
        slice(MXG, MXG + 1),
        np.r_[slice(None, j1_1g + 1), slice(j2_2g + 1, nyg)],
    )
    slices["upper_pfr_edge"] = (slice(MXG, MXG + 1), slice(j2_1g + 1, j1_2g + 1))
    slices["pfr_edge"] = (
        slice(MXG, MXG + 1),
        np.r_[
            np.r_[slice(None, j1_1g + 1), slice(j2_2g + 1, nyg)],
            slice(j2_1g + 1, j1_2g + 1),
        ],
    )

    slices["outer_midplane_a"] = (slice(None, None), int((j2_2g - j1_2g) / 2) + j1_2g)
    slices["outer_midplane_b"] = (
        slice(None, None),
        int((j2_2g - j1_2g) / 2) + j1_2g + 1,
    )

    slices["inner_midplane_a"] = (
        slice(None, None),
        int((j2_1g - j1_1g) / 2) + j1_1g + 1,
    )
    slices["inner_midplane_b"] = (slice(None, None), int((j2_1g - j1_1g) / 2) + j1_1g)

    selection = slices[name]

    return selection


# def _select_custom_core_ring(self, i):
#     """
#     Creates custom SOL ring slice within the core.
#     i = 0 is at first domain cell.
#     i = -2 is at first inner guard cell.
#     i = ixseps - MXG is the separatrix.
#     """

#     if i > self.ixseps1 - self.MXG:
#         raise Exception("i is too large!")

#     selection = (
#         slice(0 + self.MXG + i, 1 + self.MXG + i),
#         np.r_[
#             slice(self.j1_2g + 1, self.j2_2g + 1), slice(self.j1_1g + 1, self.j2_1g + 1)
#         ],
#     )

#     return self.da.isel(x=selection[0], theta=selection[1])


def _select_custom_sol_ring(ds, i, region):
    """
    Creates custom SOL ring slice beyond the separatrix.
    args[0] = i = index of SOL ring (0 is separatrix, 1 is first SOL ring)
    args[1] = region = all, inner, inner_lower, inner_upper, outer, outer_lower, outer_upper
    
    NOTE: INDEX HERE IS THE ACTUAL INDEX AS OPPOSED TO THE CUSTOM CORE RING
    
    TODO: CHECK THE OFFSETS ON X AXIS, THEY ARE POTENTIALLY WRONG
    """
    
    m = ds.metadata
    
    # if i < self.ixseps1 - self.MXG*2 :
    #     raise Exception("i is too small!")
    if i > m["nx"] - m["MXG"]*2 :
        raise Exception("i is too large!")
    
    if m["topology"] == "connected-double-null":
        
        outer_midplane_a = int((m["j2_2g"] - m["j1_2g"]) / 2) + m["j1_2g"]
        outer_midplane_b = int((m["j2_2g"] - m["j1_2g"]) / 2) + m["j1_2g"] + 1     
        # inner_midplane_a = int((m.j2_1g - m.j1_1g) / 2) + m.j1_1g 
        # inner_midplane_b = int((m.j2_1g - m.j1_1g) / 2) + m.j1_1g + 1               
        
        ny_inner = m["ny_inner"]
        MYG = m["MYG"]
        nyg = m["nyg"]
        
        # if region == "all":
        #     selection = (slice(i+1,i+2), np.r_[slice(0+.MYG, .j2_2g + 1), slice(.j1_1g + 1, self.nyg - self.MYG)])
        
        # if region == "inner":
        #     selection = (slice(i+1,i+2), slice(0+.MYG, .ny_inner + .MYG))
        # if region == "inner_lower":
        #     selection = (slice(i+1,i+2), slice(0+.MYG, inner_midplane_a +1))
        # if region == "inner_upper":
        #     selection = (slice(i+1,i+2), slice(inner_midplane_b, .ny_inner + .MYG))
        
        # if region == "outer":
        #     selection = (slice(i+1,i+2), slice(.ny_inner + .MYG*3, .nyg - .MYG))
        if region == "outer_lower":
            selection = (slice(i+1,i+2), slice(outer_midplane_b, m["nyg"] - m["MYG"]))
        elif region == "outer_upper":
            selection = (slice(i+1,i+2), slice(ny_inner + MYG*3, outer_midplane_a+1))
        else:
            raise Exception(f"Region {region} not implemented")
            
    elif m["topology"] == "single-null":
        
        if region == "all":
            selection = (slice(i+0,i+1), slice(0+MYG, nyg - MYG))
            
        else:
            raise Exception(f"Region {region} not implemented")
    
    return ds.isel(x = selection[0], theta = selection[1])
