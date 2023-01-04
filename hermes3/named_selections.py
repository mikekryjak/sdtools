import numpy as np

def select(self, ds, name, *args):
    """
    DOUBLE NULL ONLY
    Pass this touple to a field of any parameter spanning the grid
    to select points of the appropriate region.
    Each slice is a tuple: (x slice, y slice)
    Use it as: selected_array = array[slice] where slice = (x selection, y selection) = output from this method.
    """
    self = ds

    def custom_core_ring(args, self):
        """
        Creates custom SOL ring slice within the core.
        i = 0 is at first domain cell.
        i = -2 is at first inner guard cell.
        i = ixseps - MXG is the separatrix.
        """
        i = args[0]
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

    