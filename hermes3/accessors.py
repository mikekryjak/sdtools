from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
import numpy as np


@register_dataarray_accessor("hermes")
class HermesDataArrayAccessor(BoutDataArrayAccessor):
    """
    Methods on Hermes-3 datasets
    """

    def __init__(self, da):
        super().__init__(da)
        
        self.da = da
        meta = self.da.metadata

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
        
        self.j1_1 = meta["jyseps1_1"]
        self.j1_2 = meta["jyseps1_2"]
        self.j2_1 = meta["jyseps2_1"]
        self.j2_2 = meta["jyseps2_2"]
        self.ixseps2 = meta["ixseps2"]
        self.ixseps1 = meta["ixseps1"]

        self.j1_1g = self.j1_1 + self.MYG
        self.j1_2g = self.j1_2 + self.MYG * 3
        self.j2_1g = self.j2_1 + self.MYG
        self.j2_2g = self.j2_2 + self.MYG * 3
        
        
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
        
        return self.da.isel(x = selection[0], theta = selection[1])
    
    
    
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
            
            return self.da.isel(x = selection[0], theta = selection[1])