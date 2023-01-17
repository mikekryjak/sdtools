import numpy as np
from hermes3.utils import *


class Target():
    def __init__(self, case, target_name):
        self.case = case
        data = case.ds
        mass_i = constants("mass_p") * 2

        data["dr"] = data["dx"] / (data["R"] * data["Bpxy"])
        self.last = case.select_region(f"{target_name}_target")
        self.guard = case.select_region(f"{target_name}_target_guard")

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
            
        self.temperature = bndry_val("Td+").squeeze()
        self.electron_temperature = bndry_val("Te").squeeze()
        self.density = bndry_val("Ne").squeeze()
        self.ion_flux = (abs(bndry_val("NVd+")) / mass_i).squeeze()
        self.heat_flux = (gamma_i * self.temperature * constants("q_e") * self.ion_flux * 1e-6)    # MW
        self.r = np.cumsum(bndry_val("dr"))    # Length along divertor
        # width = np.insert(0,0,width)

        self.peak_temperature = self.temperature.max(axis=1)
        self.peak_density = self.density.max(axis=1)
        self.total_heat_flux = self.heat_flux.sum(axis=1) * 2*np.pi
        self.total_particle_flux = self.heat_flux.sum(axis=1) * 2*np.pi
        
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
        self.a_slice = self.case.select_custom_core_ring(self.ring_index)    # First ring
        self.b_slice = self.case.select_custom_core_ring(self.ring_index+1)    # Second ring

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



class DomainRegion():
    
    def __init__(self, case, region_name):
        self.ds = case.select_region(region_name)
        
        self.integrals = dict()

        for x in ["Sd+_src", "Sd+_iz", "Sd+_rec", "Rd+_ex", "Rd+_rec"]:
            self.integrals[x] = self.ds[x].sum(["theta", "x"])
        
    