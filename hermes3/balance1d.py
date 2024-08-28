import regex as re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from .utils import constants

class Balance1D():
    
    def __init__(
        self, 
        ds, 
        ignore_errors = False, 
        normalised = False, 
        verbose = True, 
        use_sheath_diagnostic = False,
        override_vi = False
        ):
        """
        Prepare a 1D particle and heat balance.
        
        Inputs
        ------
        ds : xarray.Dataset
            Dataset with ALL GUARD CELLS loaded in and no guard replacement
        ignore_errors : bool
            If True, ignore lack of sheath heat flux diagnostic
        normalised : bool
            If True, reproduce normalised quantities in Hermes-3 (incl. normalised mass, charge, etc).
            The dataset you provide MUST be also normalised.
        use_sheath_diagnostic : bool
            If True, use the sheath diagnostic variable for heat fluxes, otherwise use calculated value
        override_vi : bool
            If true, override Vi with Cs. Useful if runnin Hermes-3 with no_flow = True
        verbose : bool
            If True, print warnings and diagnostics
            
        """
        
        self.normalised = normalised
        self.ds = ds
        self.dom = ds.isel(pos = slice(2,-2))   # Domain
        self.terms_extracted = False
        self.tallies_extracted = False
        self.sheath_diagnostics = True
        self.use_sheath_diagnostic = use_sheath_diagnostic
        self.override_vi = override_vi
        self.verbose = verbose
        self.get_properties()  # Get settings etc
        self.reconstruct_sheath_fluxes()  # Now have .sheath dict with properties
        
        # Check for sheath diagnostic variables
        sheath_flux_candidates = [var for var in ds.data_vars if re.search("S.*\+_sheath", var)]
        if len(sheath_flux_candidates) == 0:
            self.sheath_diagnostics = False
        
        # Check if more than one time slice
        if "t" in ds.dims:
            self.time = True
        else:
            self.time = False
            
        
        
    def get_terms(self):
        """
        Extract individual balance terms based on available diagnostics
        Each term in pbal or hbal is in [s^-1] or [W]. All pressure sources
        have been converted to energy sources and everything has been integrated.
        This may cause confusion because the names of the variables are unchanged.
        """
        ds = self.ds
        dom = self.dom
        
        def integral(data):
            return (data * self.dom["dv"]).sum("pos").values

        ### Particle balance
        pbal = {}
        pbalsum = {}
        for param in ["Sd+_src", "Sd_src", "Sd+_feedback", "Sd+_iz", "Sd+_rec", "Sd_target_recycle", "Sd+_sheath"]:
            if param in ds:
                pbal[param] = integral(dom[param])
            else:
                if self.verbose:
                    print(f"Warning! {param} not found, results may be incorrect")
                pbal[param] = np.zeros_like(integral(dom["Ne"]))
       
                
        ### Heat balance
        hbal = {}
        hbalsum = {}
        for param in ["Pd+_src", "Pe_src", "Pd_src", "Rd+_ex", "Rd+_rec", "Rar", "Ed_target_recycle", "Ee_sheath", "Ed+_sheath"]:
            if param in ds:
                hbal[param] = integral(dom[param])*1e-6   # MW

                if param.startswith("P"):
                    newname = "E" + param[1:]
                    hbal[newname] = hbal[param] * 3/2    # Convert from pressure to energy
                    del hbal[param]
                    
                if param == "Rar":
                    hbal[param] = -1 * abs(hbal[param])   # Correct inconsistency in sign convention
            else:
                if self.verbose:
                    print(f"Warning! {param} not found, results may be incorrect")
                hbal[param] = np.zeros_like(integral(dom["Ne"]))
                
        if self.use_sheath_diagnostic is False or self.sheath_diagnostics is False:
            if self.verbose: print("|||WARNING: Overwriting sheath diagnostics with calculated values")
            hbal["Ee_sheath"] = self.sheath["hfe"]
            hbal["Ed+_sheath"] = self.sheath["hfi"]
            pbal["Sd+_sheath"] = self.sheath["pfi"]
        
        self.pbal = pbal
        self.hbal = hbal
        
        if self.sheath_diagnostics is False:
            if self.verbose: print("Sheath diagnostics not available, attempting to reconstruct...")
            self.reconstruct_sheath_fluxes()
            
        self.terms_extracted = True
        
    def get_tallies(self):
        """
        Derive tallies from the balance terms which are useful in preparing a balance
        """
        ds = self.ds
        
        ## Particle balance
        pbal = self.pbal
        pbalsum = {}
        pbalsum["sources_sum"] = pbal["Sd+_src"] + pbal["Sd_src"] + pbal["Sd+_feedback"]
        
        # pbalsum["net_iz"] = pbal["Sd+_iz"] + pbal["Sd+_rec"]
        pbalsum["sheath_sum"] = pbal["Sd+_sheath"] + pbal["Sd_target_recycle"]
        pbalsum["recycle_frac"] = abs(pbal["Sd_target_recycle"] / pbal["Sd+_sheath"])
        
        pbalerror = {}
        pbalerror["imbalance"] = pbalsum["sources_sum"] + pbalsum["sheath_sum"]
        
        ## Heat balance
        hbal = self.hbal
        hbalsum = {}
        
        hbalsum["sources_sum"] = 0
        for param in hbal:
            if param.endswith("src"):
                hbalsum["sources_sum"] += hbal[param]
    
        hbalsum["R_hydr_sum"] = hbal["Rd+_ex"] + hbal["Rd+_rec"]
        hbalsum["R_imp_sum"] = hbal["Rar"] if "Rar" in ds else np.zeros_like(hbal["Ed+_src"])
        hbalsum["R_sum"] = hbalsum["R_hydr_sum"] + hbalsum["R_imp_sum"]
        hbalsum["R_and_sources_sum"] = hbalsum["sources_sum"] + hbalsum["R_sum"]
        hbalsum["sheath_sum"] = hbal["Ed+_sheath"] + hbal["Ee_sheath"]
        
        hbalerror = {}
        hbalerror["error"] = (hbalsum["R_and_sources_sum"] + hbalsum["sheath_sum"]) / hbalsum["sources_sum"]
        
        self.pbalsum = pbalsum
        self.hbalsum = hbalsum
        self.pbalerror = pbalerror
        self.hbalerror = hbalerror
        self.tallies_extracted = True
        
    def get_target_value(self, data):
        ds = self.ds
        if ds.metadata["MYG"] == 0:
            raise Exception("Dataset must be read in with all guard cells")
        return (data.isel(pos=-3) + data.isel(pos=-2))/2
    
    def get_properties(self):
        """
        Get sheath BC settings if sheath_boundary_simple is used
        Set constants to appropriate values depending on normalisation
        """
        ds = self.ds

        if "sheath_boundary_simple" not in ds.options.keys():
            raise Exception("Extracting gamma only implemented from sheath_boundary_simple")
        
        if "gamma_i" in ds.options["sheath_boundary_simple"].keys():
            self.gamma_i = ds.options["sheath_boundary_simple"]["gamma_i"]
        else:
            self.gamma_i = 3.5
            
        if "gamma_e" in ds.options["sheath_boundary_simple"].keys():
            self.gamma_e = ds.options["sheath_boundary_simple"]["gamma_e"]
        else:
            self.gamma_e = 3.5
            
    
        self.Zi = ds.options["d+"]["charge"]
        self.Mi = ds.options["d+"]["AA"] * constants("mass_p")
        self.Me = constants("mass_e")
        self.qe = constants("q_e")
        
        if self.normalised:
            self.Mi /= constants("mass_p")
            self.Me /= constants("mass_p")
            self.qe /= constants("q_e")

        if "sheath_ion_polytropic" in ds.options["sheath_boundary_simple"].keys():
            self.sheath_ion_polytropic = ds.options["sheath_boundary_simple"]["sheath_ion_polytropic"]
        else:
            self.sheath_ion_polytropic = 1
            
        if "secondary_electron_coef" in ds.options["sheath_boundary_simple"].keys():
            self.Ge = ds.options["sheath_boundary_simple"]["secondary_electron_coef"]
        else:
            self.Ge = 0
            
        if "phi_wall" in ds.options["sheath_boundary_simple"].keys():
            self.phi_wall = ds.options["sheath_boundary_simple"]["phi_wall"]
        else:
            self.phi_wall = 0
            
            
    def reconstruct_sheath_fluxes(self, override_vi = False):
        """
        Calculate sheath fluxes from the fields without need for diagnostic variable
        
        Inputs
        ------
        
        """
        
        ds = self.ds
        qe = self.qe
        Zi = self.Zi
        Mi = self.Mi
        Me = self.Me
        Ge = self.Ge
        sheath_ion_polytropic = self.sheath_ion_polytropic
        phi_wall = self.phi_wall
        
        ones_template = np.ones_like(ds["Ne"].isel(pos=-3))

        dasheath = np.array(ones_template * self.get_target_value(ds["da"]).values)
        dv = ones_template * ds.isel(pos=-3)["dv"].values

        visheath = self.get_target_value(ds["Vd+"]).values
        vesheath = self.get_target_value(ds["Ve"]).values

        nesheath = self.get_target_value(ds["Ne"]).values
        nisheath = self.get_target_value(ds["Nd+"]).values

        tesheath = self.get_target_value(ds["Te"]).values
        tisheath = self.get_target_value(ds["Td+"]).values

        cssheath = np.sqrt((sheath_ion_polytropic * tisheath*qe + Zi * tesheath*qe) / Mi)   # [m/s] Bohm criterion sound speed

        ion_sum = Zi * nisheath * cssheath   # Sheath current
        phisheath = tesheath * np.log(np.sqrt(tesheath / (Me * 2*np.pi)) * (1 - Ge) * nesheath / ion_sum)  # [V] sheath potential, (note Neumann BC)
        vesheath = np.sqrt(tesheath / (2*np.pi * Me)) * (1 - Ge) * np.exp(-(phisheath - phi_wall)/tesheath)
        
        if self.override_vi == True:
            visheath = cssheath

        pfi_sheath = -1 * visheath * nisheath * dasheath   # [s^-1] ion particle flux into domain 
        pfe_sheath = -1 * vesheath * nesheath * dasheath   # [s^-1] electron particle flux into domain 

        hfi_sheath = pfi_sheath * self.gamma_i * tisheath*qe * 1e-6   # [MW] electron heat flux into domain
        hfe_sheath = pfe_sheath * self.gamma_e * tesheath*qe * 1e-6   # [MW] electron heat flux into domain
        
        self.sheath = dict(
            da = dasheath,
            ion_sum = ion_sum, 
            phi = phisheath,
            vi = visheath, 
            ve = vesheath, 
            ne = nesheath, 
            ni = nisheath, 
            te = tesheath, 
            ti = tisheath, 
            cs = cssheath, 
            pfi = pfi_sheath,
            pfe = pfe_sheath,
            hfe = hfe_sheath,
            hfi = hfi_sheath,
            )
        
    def print_sheath_conditions(self):
        s = self.sheath
        
        # Return final index if have multiple time slices
        for param in s.keys():
            if self.time is True:
                s[param] = s[param][-1]

        print(f'dasheath = {s["da"]:.6}')
        print(f'nesheath = {s["ne"]:.6}')
        print(f'nisheath = {s["ni"]:.6}')
        print(f'tesheath = {s["te"]:.6}')
        print(f'tisheath = {s["ti"]:.6}')
        print(f'cssheath = {s["cs"]:.6}')
        print(f'vesheath = {s["ve"]:.6}')
        print(f'visheath = {s["vi"]:.6}')
        
        
    def plot_heat_balance(self):
        fig, ax = plt.subplots()

        pos = {}
        neg = {}

        for key in self.hbal:
            
            # Find last value if time series
            if self.time:
                val = self.hbal[key][-1]
            else:
                val = self.hbal[key]
            
            # Sort into sources and sinks
            if val > 0:
                pos[key] = self.hbal[key]
            elif val < 0:
                neg[key] = self.hbal[key]
            else:
                if self.verbose: print(f"{key} is zero, dropping")
                
                
        if "t" not in self.ds.dims:
            raise Exception("Heat balance plot only available for time series data")
        
        sum_pos = np.sum(list(pos.values()), axis = 0)
        sum_neg = np.sum(list(neg.values()), axis = 0)
        largest_flux = np.max([sum_neg[-1], sum_pos[-1]])
        
        t = self.ds.t
        ax.stackplot(t, list(pos.values()), labels = pos.keys(), baseline = "zero", alpha = 0.7)
        ax.stackplot(t, list(neg.values()), labels = neg.keys(), baseline = "zero", alpha = 0.7)
        ax.hlines(0, t[0], t[-1], color = "k", linestyle = "--")
        
        ax.hlines(largest_flux, t[0], t[-1], color = "k", linestyle = ":")
        ax.hlines(largest_flux*-1, t[0], t[-1], color = "k", linestyle = ":")

        ax.legend(loc = "upper left", bbox_to_anchor = (1,1))
        ax.set_title("Domain heat balance history")
        ax.set_ylabel("[MW]")
        ax.set_xlabel("Time [s]")
        
    def plot_flux_balance(self, 
                          use_diagnostics = True,
                          xlims = (None, None),
                          flux_style = {}):

        ds = self.ds.isel(pos=slice(2,-2))
        if "t" in ds.dims: ds = ds.isel(t=-1)

        src_i = ((ds["Pd+_src"]) * ds["dv"]).cumsum("pos") * 1e-6 * 3/2
        src_e = ((ds["Pe_src"]) * ds["dv"]).cumsum("pos") * 1e-6 * 3/2
        src_tot = src_i + src_e

        if use_diagnostics is True:
            if "div_cond_par_e" in ds.data_vars:
                hfi_cond = (ds["div_cond_par_d+"] * ds["dv"]).cumsum("pos") * -1e-6 
                hfe_cond = (ds["div_cond_par_e"] * ds["dv"]).cumsum("pos") * -1e-6 
            else:
                raise Exception("Conduction divergence diagnostics not found")
        
        else:
            hfe_cond = ds["kappa_par_e"] * np.gradient(ds["Te"], ds["pos"]) * ds["da"] * -1e-6 #* 1e-2
            hfi_cond = ds["kappa_par_d+"] * np.gradient(ds["Td+"], ds["pos"]) * ds["da"] * -1e-6 #* 1e-2
        
        hfe_p_adv = 5/2 * ds["Ve"] * ds["Ne"] * ds["Te"]*self.qe * 1e-6
        hfe_k_adv = (0.5 * ds["Nd+"] * self.Me * ds["Ve"]**2) * ds["Ve"] * 1e-6
        
        hfi_p_adv = 5/2 * ds["Vd+"] * ds["Nd+"] * ds["Td+"]*self.qe * 1e-6
        hfi_k_adv = (0.5 * ds["Nd+"] * self.Mi * ds["Vd+"]**2) * ds["Vd+"] * 1e-6
        
        hfi_conv = ds["Vd+"] * (ds["Pd+"] + ds["Pe"]) * ds["da"] * 5/2  * 1e-6
        hf_tot = hfe_cond + hfi_cond + hfi_conv + hfe_p_adv + hfe_k_adv + hfi_p_adv + hfi_k_adv
        
        rad = []
        for param in ds:
            if param.startswith("R"):
                rad.append(ds[param] * ds["dv"])
                
        if len(rad) > 0:
            rad = np.sum(rad, axis = 0) * 1e-6  # All channel summed
            rad = np.cumsum(rad * ds["dv"])     # Cumulative integral
        else:
            rad = np.zeros_like(ds["Vd+"])

        fig, ax = plt.subplots()
        style_src = dict(lw = 1, ls = "--")
        ax.plot(ds["pos"], hfe_cond, c = "indigo", label = "electron conduction", **flux_style)
        ax.plot(ds["pos"], hfi_cond, c = "tomato", label = "ion conduction", **flux_style)
        ax.plot(ds["pos"], hfi_p_adv, c = "deeppink", label = "ion internal energy adv.", **flux_style)
        ax.plot(ds["pos"], hfe_p_adv, c = "skyblue", label = "electron internal energy adv.", **flux_style)
        
        ax.plot(ds["pos"], hfi_k_adv, c = "deeppink", label = "ion kinetic energy adv.", ls = "--", **flux_style)
        ax.plot(ds["pos"], hfe_k_adv, c = "skyblue", label = "electron kinetic energy adv.", ls = "--", **flux_style)
        
        ax.plot(ds["pos"], rad, c = "gold", label = "Total radiation", **flux_style)
        ax.plot(ds["pos"], hf_tot, c = "grey", label = "Total flux", alpha = 0.3, lw = 4)
        ax.plot(ds["pos"], src_tot, label = "Total source", c = "grey", alpha = 0.3, lw = 4, ls = ":")
        ax.legend(loc = "upper center", bbox_to_anchor=(0.5,-0.15), ncols = 2)
        ax.set_xlabel("Lpar [m]")
        ax.set_ylabel("[MW]")
        ax.set_title(f"Heat flux balance")
        
        if xlims != (None, None):
            ax.set_xlim(xlims)
            
        # ax.set_yscale("log")
        
    def print_balances2(self):
        """
        Print in a way to compare Ben's balance script
        """
        pbal = self.pbal
        hbal = self.hbal
        sheath = self.sheath
        dom = self.dom
        
        ## If more than one time slice, select final one
        if len(pbal) > 0:
            for param in pbal:
                pbal[param] = pbal[param][-1]
            for param in hbal:
                hbal[param] = hbal[param][-1]
            for param in sheath:
                if param not in ["da"]:
                    sheath[param] = sheath[param][-1]

        def integral(data):
            return (data * self.dom["dv"]).sum("pos").values
        
        if "t" in dom.dims: dom = dom.isel(t=-1)
        total_volume = dom['dv'].sum()
        area_sheath = self.get_target_value(dom['da'])
        
        print("\n\n")
        
        print("Simulation geometry")
        print(f"    Volume: {total_volume} m^3")
        print(f"    Area: {area_sheath} m^2")
        print(f"    Volume / area: {total_volume / area_sheath} m")
        print("")
        
        
        ###########################################################
        # Particle balance
        
        
        particle_source = self.pbal["Sd+_src"]
        ionization = self.pbal["Sd+_iz"]
        recombination = self.pbal["Sd+_rec"]
        total_ion_source = integral(dom["SNd+"])
        feedback_source = self.pbal["Sd+_feedback"]

        print(f"Total ion particle source:  {total_ion_source}  (check: {particle_source + ionization + recombination + feedback_source}")
        print(f"  |- External ion source:   {particle_source}")
        print(f"  |- Feedback source:       {feedback_source}")
        print(f"  |- Ionization source:     {ionization}")
        print(f"  |- Recombination source:  {recombination}")
        print("")
        
        
        
        total_neutral_source = integral(dom["SNd"])
        neutral_recycle = integral(dom["Sd_target_recycle"])
        neutral_source = integral(dom["Sd_src"])

        print(f"Total neutral particle source: {total_neutral_source} (check: {neutral_source - ionization - recombination + neutral_recycle})")
        print(f"  |- External neutral source:  {neutral_source}")
        print(f"  |- Target recycling:         {neutral_recycle}")
        print(f"  |- Ionization source:        {-ionization}")
        print(f"  |- Recombination source:     {-recombination}")
        print("")
        
        
        print(f"Sheath")
        print(f"  Density:     {sheath['ni']} m^-3")
        print(f"  Sound speed: {sheath['cs']} m/s")
        print(f"  Flow speed:  {sheath['vi']} m/s")
        print(f"  Ion sink:    {pbal['Sd+_sheath']} s^-1")
        print(f"  Neutral recycling: {neutral_recycle} s^-1")
        print("")
        
        
        ###########################################################
        
        ## Pressure sources have already been converted to energy, no need for 3/2 here
        ion_heating = hbal["Ed+_src"]
        electron_heating = hbal["Ee_src"]

        print(f"Total input power:     {(ion_heating + electron_heating) * 1e-6} MW")
        print(f"  |- Ion heating:      {ion_heating * 1e-6} MW")
        print(f"  |- Electron heating: {electron_heating * 1e-6} MW")
        print("")
        
        recycle_heating = hbal["Ed_target_recycle"]
        ion_energy_flux = hbal["Ed+_sheath"]
        electron_energy_flux = hbal["Ee_sheath"]
        
        ion_convection = 2.5 * sheath["ni"] * sheath["ti"] * sheath["vi"] * sheath["da"]
        ion_kinetic = 0.5 * sheath["vi"] * self.Mi * sheath["ni"] * sheath["da"]
        
        electron_convection = 2.5 * sheath["ne"] * sheath["te"] * sheath["ve"] * sheath["da"]
        electron_kinetic = 0.5 * sheath["ve"] * self.Me * sheath["ne"] * sheath["da"]
        
        R_rec = hbal["Rd+_rec"]
        R_ex = hbal["Rd+_ex"]

        print(f"Total power loss: {(ion_energy_flux + electron_energy_flux - recycle_heating - R_ex - R_rec) * 1e-6} MW")
        print(f"  |- Ions:              {ion_energy_flux * 1e-6} MW")
        print(f"      |- Convection          {ion_convection * 1e-6} MW")
        print(f"      |- Kinetic energy      {ion_kinetic * 1e-6} MW")
        print(f"      |- Conduction          {(ion_energy_flux - ion_kinetic - ion_convection) * 1e-6} MW")
        print(f"  |- Electrons:         {electron_energy_flux * 1e-6} MW")
        print(f"      |- Convection          {electron_convection * 1e-6} MW")
        print(f"      |- Kinetic energy      {electron_kinetic * 1e-6} MW")
        print(f"      |- Conduction          {(electron_energy_flux - electron_kinetic - electron_convection) * 1e-6} MW")
        print(f"  |- Recycled neutrals: {-recycle_heating * 1e-6} MW")
        print(f"  |- Ionization:        {-R_ex * 1e-6} MW")
        print(f"  |- Recombination:     {-R_rec * 1e-6} MW")
        print("")

        
    def print_balances(self):
        pbal = self.pbal
        hbal = self.hbal
        pbalsum = self.pbalsum
        hbalsum = self.hbalsum
        pbalerror = self.pbalerror
        hbalerror = self.hbalerror
        

        
        ### Print output for particle balance
        print("\nParticle flows [s^-1]")
        print("----------------------")
        for param in pbal:
            data = pbal[param][-1] if self.time else pbal[param]
            print(f"{param}: {data:.3e}")
            
        print("----------------------")
        print("Derived quantities")
        print("----------------------")
        for param in pbalsum:
            data = pbalsum[param][-1] if self.time else pbalsum[param]
            print(f"{param}: {data:.3e}")
            
        print("----------------------")
        print("Diagnostic output")
        print("----------------------")
        error = pbalerror["imbalance"][-1] if self.time else pbalerror["imbalance"]
        print(f"Imbalance: sources + net sheath flux: {error:.3e} [s^-1]")

        ### Print output for heat balance
        print("\n\nHeat flows [MW]")
        # print("Check recycle fraction against expectations")
        print("----------------------")
        for param in hbal:
            data = hbal[param][-1] if self.time else hbal[param]
            print(f"{param}: {data:.3e}")

        print("----------------------")
        print("Derived quantities")
        print("----------------------")
        for param in hbalsum:
            data = hbalsum[param][-1] if self.time else hbalsum[param]
            print(f"{param}: {data:.3e}")
            
        print("----------------------")
        print("Diagnostic output")
        print("----------------------")
        error = hbalerror["error"][-1] if self.time else hbalerror["error"]
        print(f"Error: (R + Src + Sheath) / Src: {error:.2%}")