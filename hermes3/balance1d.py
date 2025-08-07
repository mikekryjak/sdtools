import regex as re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
from .utils import constants

class Balance1D():
    
    def __init__(
        self, 
        ds, 
        ignore_errors = False, 
        normalised = False, 
        verbose = True, 
        use_sheath_diagnostic = False,
        override_vi = False,
        sheath_flux_method = "flux_diagnostic",
        impurity_name = "Ar",
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
        self.impurity_name = impurity_name
        self.dom = ds.isel(pos = slice(2,-2))   # Domain
        self.terms_extracted = False
        self.tallies_extracted = False
        
        self.use_sheath_diagnostic = use_sheath_diagnostic
        self.override_vi = override_vi
        self.verbose = verbose
        self.get_properties()  # Get settings etc
        self.calculate_sheath_fluxes(method = sheath_flux_method)  # Now have .sheath dict with properties
        
        # # Check for sheath diagnostic variables
        # self.sheath_diagnostics = True
        # sheath_flux_candidates = [var for var in ds.data_vars if re.search("S.*\\+_sheath", var)]
        # if len(sheath_flux_candidates) == 0:
        #     self.sheath_diagnostics = False
        
        # Check if more than one time slice
        if "t" in ds.dims:
            self.time = True
        else:
            self.time = False
            
        # Calculate the balance
        # self.get_terms()
        # self.get_tallies()
            
        
        
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
        for param in ["Sd+_src", "Sd_src", "Sd+_feedback", "Sd+_iz", "Sd+_rec", "Sd_target_recycle"]:
            if param in ds:
                pbal[param] = integral(dom[param])
            else:
                if self.verbose:
                    print(f"Warning! {param} not found, results may be incorrect")
                pbal[param] = np.zeros_like(integral(dom["Ne"]))
       
                
        ### Heat balance
        hbal = {}
        hbalsum = {}
        for param in ["Pd+_src", "Pe_src", "Pd_src", "Rd+_ex", "Rd+_rec", f"R{self.impurity_name}", "Ed_target_recycle"]:
            if param in ds:
                hbal[param] = integral(dom[param])*1e-6   # MW

                if param.startswith("P"):
                    newname = "E" + param[1:]
                    hbal[newname] = hbal[param] * 3/2    # Convert from pressure to energy
                    del hbal[param]
                    
                if param == f"R{self.impurity_name}":
                    hbal[param] = -1 * abs(hbal[param])   # Correct inconsistency in sign convention
            else:
                if self.verbose:
                    print(f"Warning! {param} not found, results may be incorrect")
                hbal[param] = np.zeros_like(integral(dom["Ne"]))
                
        hbal["Ee_sheath"] = self.sheath["hfe"]
        hbal["Ed+_sheath"] = self.sheath["hfi"]
        pbal["Sd+_sheath"] = self.sheath["pfi"]
        self.pbal = pbal
        self.hbal = hbal
    
            
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
        pbalerror["imbalance"] = (pbalsum["sources_sum"] - pbal["Sd_target_recycle"]) / pbalsum["sources_sum"]
        
        ## Heat balance
        hbal = self.hbal
        hbalsum = {}
        
        hbalsum["sources_sum"] = 0
        for param in hbal:
            if param.endswith("src"):
                hbalsum["sources_sum"] += hbal[param]
    
        hbalsum["R_hydr_sum"] = hbal["Rd+_ex"] + hbal["Rd+_rec"]
        hbalsum["R_imp_sum"] = hbal[f"R{self.impurity_name}"] if f"R{self.impurity_name}" in ds else np.zeros_like(hbal["Ed+_src"])
        hbalsum["R_sum"] = hbalsum["R_hydr_sum"] + hbalsum["R_imp_sum"]
        hbalsum["R_and_sources_sum"] = hbalsum["sources_sum"] + hbalsum["R_sum"]
        hbalsum["sheath_sum"] = hbal["Ed+_sheath"] + hbal["Ee_sheath"]
        hbalsum["power_in"] = hbalsum["sources_sum"]
        hbalsum["power_out"] = hbalsum["R_sum"] + hbalsum["sheath_sum"]
        
        hbalerror = {}
        hbalerror["error"] = (hbalsum["R_and_sources_sum"] - hbalsum["sheath_sum"]) / hbalsum["sources_sum"]
        
        self.pbalsum = pbalsum
        self.hbalsum = hbalsum
        self.pbalerror = pbalerror
        self.hbalerror = hbalerror
        self.tallies_extracted = True
        
    def get_target_value(self, data):
        ds = self.ds
        if ds.metadata["MYG"] == 0:
            raise Exception("Dataset must be read in with all guard cells")
        return ((data.isel(pos=-3) + data.isel(pos=-2))/2)
    
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
            
            
    def get_sheath_fluxes(self):
        """
        Calculate sheath fluxes from flux diagnostic variables
        """
        
        
    def calculate_sheath_fluxes(self,
                                method = "flux_diagnostic",
                                override_vi = False,
                                calculate_ve = False
                                  ):
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

        if calculate_ve:
            ion_sum = Zi * nisheath * cssheath   # Sheath current
            phisheath = tesheath * np.log(np.sqrt(tesheath / (Me * 2*np.pi)) * (1 - Ge) * nesheath / ion_sum)  # [V] sheath potential, (note Neumann BC)
            vesheath = np.sqrt(tesheath / (2*np.pi * Me)) * (1 - Ge) * np.exp(-(phisheath - phi_wall)/tesheath)
        else:
            ion_sum = 0
            phisheath = 0
            
        
        if self.override_vi == True:
            visheath = cssheath

        if method == "reconstruction":
            pfi_sheath = -1 * visheath * nisheath * dasheath   # [s^-1] ion particle flux into domain 
            pfe_sheath = -1 * vesheath * nesheath * dasheath   # [s^-1] electron particle flux into domain 

            hfi_sheath = pfi_sheath * self.gamma_i * tisheath*qe * 1e-6   # [MW] electron heat flux into domain
            hfe_sheath = pfe_sheath * self.gamma_e * tesheath*qe * 1e-6   # [MW] electron heat flux into domain
            
        elif method == "flux_diagnostic":
            pfi_sheath = ds["pfd+_tot_ylow"].isel(pos=-2)
            pfe_sheath = pfi_sheath
            hfe_sheath = ds["efe_tot_ylow"].isel(pos=-2) * 1e-6
            hfi_sheath = ds["efd+_tot_ylow"].isel(pos=-2) * 1e-6
            
        elif method == "source_diagnostic":
            pfi_sheath = (ds["Sd+_sheath"] * ds["dv"]).values.sum()
            pfe_sheath = pfi_sheath
            hfi_sheath = (ds["Ed+_sheath"] * ds["dv"]).values.sum() * 1e-6
            hfe_sheath = (ds["Ee_sheath"] * ds["dv"]).values.sum() * 1e-6
            
        
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
        
    
        
    def plot_flux_balance_old(self, 
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
        
    def plot_flux_balance2_old(self, xlims = (None, None), title = ""):
        """
        This comes from Ben
        """
        
        print("SOME OF THESE ARE INCORRECT")
        ds = self.ds
        if "t" in ds.dims: ds = ds.isel(t=-1)
        

        if len(sys.argv) != 2:
            print("Usage: {} path".format(sys.argv[0]))
            sys.exit(1)

        path = sys.argv[1]

        # These diagnostics describe parallel flows of energy
        # flows = [
        #     ("KineticFlow_d+_ylow", "ion kinetic"),
        #     ("KineticFlow_e_ylow", "electron kinetic"),
        #     ("KineticFlow_d_ylow", "neutral kinetic"),
        #     ("ConductionFlow_d+_ylow", "ion conduction"),
        #     ("ConductionFlow_e_ylow", "electron conduction"),
        #     ("ConductionFlow_d_ylow", "neutral conduction"),
        #     ("EnergyFlow_d+_ylow", "ion total"),
        #     ("EnergyFlow_e_ylow", "electron total"),
        #     ("EnergyFlow_d_ylow", "neutral total"),
        # ]
        
        flows = [
            ("efd+_kin_ylow", "ion kinetic"),
            ("efe_kin_ylow", "electron kinetic"),
            ("efd_kin_ylow", "neutral kinetic"),
            ("efd+_cond_ylow", "ion conduction"),
            ("efe_cond_ylow", "electron conduction"),
            ("efd_cond_ylow", "neutral conduction"),
            ("efd+_tot_ylow", "ion total"),
            ("efe_tot_ylow", "electron total"),
            ("efd_tot_ylow", "neutral total"),
        ]


        # Sources of energy. To be summed along field line
        sources = [
            ("Pd+_src", "Ion external source", 3./2),
            ("Pe_src", "Electron external source", 3./2),
            ("Rd+_ex", "Excitation radiation", 1.0),
            ("Rd+_rec", "Recombination losses", 1.0),
            #("Edd+_cx", "CX d -> d+", -1.0),
            #("Ed+_iz", "IZ d -> d+", 1.0),
            #("SPd+", "SPd+", 3./2),
            #("SPe", "SPe", 3./2),
        ]


        dV = ds["dv"].values[2:-2]
            
        def remove_guards(var1D):
            return var1D[2:-1]

        flow_data = {}
        total_flow = np.zeros_like(remove_guards(ds["efd+_kin_ylow"].values * 1e-6))
        
        for name, label in flows:
            if name in ds:
                value = ds[name].values
            else:
                print(f"Flow {name} not found")
                continue
            # Convert to MW
            this_flow = remove_guards(value * 1e-6)
            flow_data[name] = (this_flow,
                            label)
            if name.startswith("ef"):
            # if name.startswith("E"):
                # Energyflow
                total_flow += this_flow
        print(total_flow.shape)

        flow_data["total"] = (total_flow,
                            "Total flow")

        source_data = {}
        total_source = 0.0
        for name, label, factor in sources:
            if name in ds:
                value = ds[name].values
            else:
                print(f"Source {name} not found")
                continue
                
            value *= factor * 1e-6

            value[2:-2] *= dV
            value = value[1:-2]
            value[0] = 0.0 # No influx from lower boundary
            this_source = np.cumsum(value)
            source_data[name] = (this_source,
                                label)
            total_source += this_source
        source_data["total"] = (total_source,
                                "Total sources")

        fig, ax = plt.subplots(dpi = 300)
        total_style = dict(lw = 3, alpha = 0.5)
        for name, value in flow_data.items():
            style = total_style if "total" in name else {}
            ax.plot(ds["pos"].values[2:-1], value[0], label = value[1], **style)

        for name, value in source_data.items():
            style = total_style if "total" in name else {}
            ax.plot(ds["pos"].values[2:-1], value[0], label = value[1], linestyle='--', **style)
            
        ax.legend(loc = "upper left", bbox_to_anchor = (1,1))
        ax.set_ylabel('MW')
        
        if xlims != (None, None):
            ax.set_xlim(xlims)
        ax.set_xlabel("Pos [m]")
        ax.set_title(title)

        
    def print_balances2(self):
        """
        Print in a way to compare Ben's balance script
        """
        pbal = self.pbal.copy()
        hbal = self.hbal.copy()
        sheath = self.sheath.copy()
        dom = self.dom
        if "t" in dom.dims: dom = dom.isel(t=-1)
                
        
        # If more than one time slice, select final one
        if self.time:
            for param in pbal:
                pbal[param] = pbal[param][-1]
            for param in hbal:
                hbal[param] = hbal[param][-1]
            for param in sheath:
                # if param not in ["da"]:
                sheath[param] = sheath[param][-1]

        def integral(data):
            return (data * self.dom["dv"]).sum("pos").values
        
        
        total_volume = dom['dv'].sum().values
        area_sheath = self.get_target_value(dom['da']).values
        
        print("\n\n")
        
        print("Simulation geometry")
        print(f"    Volume: {total_volume} m^3")
        print(f"    Area: {area_sheath} m^2")
        print(f"    Volume / area: {total_volume / area_sheath} m")
        print("")
        
        
        ###########################################################
        # Particle balance
        
        
        particle_source = pbal["Sd+_src"]
        ionization = pbal["Sd+_iz"]
        recombination = pbal["Sd+_rec"]
        total_ion_source = integral(dom["SNd+"])
        feedback_source = pbal["Sd+_feedback"]

        print(f"Total ion particle source:  {total_ion_source:.3e}  (check: {particle_source + ionization + recombination + feedback_source:.3e})")
        print(f"  |- External ion source:   {particle_source:.3e}")
        print(f"  |- Feedback source:       {feedback_source:.3e}")
        print(f"  |- Ionization source:     {ionization:.3e}")
        print(f"  |- Recombination source:  {recombination:.3e}")
        print("")
        
        
        
        total_neutral_source = integral(dom["SNd"])
        neutral_recycle = integral(dom["Sd_target_recycle"])
        neutral_source = integral(dom["Sd_src"])

        print(f"Total neutral particle source: {total_neutral_source:.3e} (check: {neutral_source - ionization - recombination + neutral_recycle:.3e})")
        print(f"  |- External neutral source:  {neutral_source:.3e}")
        print(f"  |- Target recycling:         {neutral_recycle:.3e}")
        print(f"  |- Ionization source:        {-ionization:.3e}")
        print(f"  |- Recombination source:     {-recombination:.3e}")
        print("")
        
        
        print(f"Sheath")
        print(f"  Density:     {sheath['ni']:.3e} m^-3")
        print(f"  Sound speed: {sheath['cs']:.3e} m/s")
        print(f"  Flow speed:  {sheath['vi']:.3e} m/s")
        print(f"  Ion sink:    {pbal['Sd+_sheath']:.3e} s^-1")
        print(f"  Neutral recycling: {neutral_recycle:.3e} s^-1")
        print("")
        
        
        ###########################################################
        
        ## Pressure sources have already been converted to energy, no need for 3/2 here
        ion_heating = hbal["Ed+_src"]
        electron_heating = hbal["Ee_src"]

        print(f"Total input power:     {(ion_heating + electron_heating):.3e} MW")
        print(f"  |- Ion heating:      {ion_heating:.3e} MW")
        print(f"  |- Electron heating: {electron_heating:.3e} MW")
        print("")
        
        recycle_heating = hbal["Ed_target_recycle"]
        ion_energy_flux = hbal["Ed+_sheath"]
        electron_energy_flux = hbal["Ee_sheath"]
        
        ion_convection = 2.5 * sheath["ni"] * sheath["ti"] * sheath["vi"] * sheath["da"] * constants("q_e")
        ion_kinetic = 0.5 * sheath["vi"] * self.Mi * sheath["ni"] * sheath["da"]
        
        electron_convection = 2.5 * sheath["ne"] * sheath["te"] * sheath["ve"] * sheath["da"] * constants("q_e")
        electron_kinetic = 0.5 * sheath["ve"] * self.Me * sheath["ne"] * sheath["da"]
        
        R_rec = hbal["Rd+_rec"]
        R_ex = hbal["Rd+_ex"]

        print(f"Total power loss: {(ion_energy_flux + electron_energy_flux - recycle_heating - R_ex - R_rec):.3e} MW")
        print(f"  |- Ions:              {ion_energy_flux:.3e} MW")
        print(f"      |- Convection          {ion_convection:.3e} MW")
        print(f"      |- Kinetic energy      {ion_kinetic:.3e} MW")
        print(f"      |- Conduction          {(ion_energy_flux - ion_kinetic - ion_convection):.3e} MW")
        print(f"  |- Electrons:         {electron_energy_flux:.3e} MW")
        print(f"      |- Convection          {electron_convection:.3e} MW")
        print(f"      |- Kinetic energy      {electron_kinetic:.3e} MW")
        print(f"      |- Conduction          {(electron_energy_flux - electron_kinetic - electron_convection):.3e} MW")
        print(f"  |- Recycled neutrals: {-recycle_heating:.3e} MW")
        print(f"  |- Ionization:        {-R_ex:.3e} MW")
        print(f"  |- Recombination:     {-R_rec:.3e} MW")
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
        print(f"(sources + recycled flow) / sources: {error:.3e}")

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
        print(f"Error: (R + Src - Sheath) / Src: {error:.2%}")
        
    def check_bug(self):
        if type(self.pbal["Sd+_src"]) == np.float64:
            raise Exception("BUG FOUND")
        else:
            print("No bug found")
            
            
class FluxBalance():
    
    def __init__(self, ds, include_neutrals = True):
        self.ds = ds
        self.include_neutrals = include_neutrals
        if "t" in ds.coords:
            ds = self.ds = ds.isel(t=-1)
            
        dom = self.dom = ds.isel(pos = slice(2,-1))  # Remove guard cells
        domp = self.domp = dom.shift(pos = -1)  # # Shift to get the next point in the domain
        self.xaxis = dom["pos"].values.max() - dom["pos"].values + 1e-3
    
    def plot_total_balance(self):
        
        fig, ax = plt.subplots(figsize = (10, 4))

        dom = self.dom
        domp = self.domp
        x = self.xaxis


        ## Divergences
        # flow_LHS - flow_RHS
        electrons = dom[f"efe_tot_ylow"] - domp[f"efe_tot_ylow"]
        ions = dom[f"efd+_tot_ylow"] - domp[f"efd+_tot_ylow"]
        if self.include_neutrals:
            neutrals = dom[f"efd_tot_ylow"] - domp[f"efd_tot_ylow"]
            radiation = (abs(dom["Rd+_ex"]) + abs(dom["Rd+_rec"]))*dom["dv"] * -1
            radiation[-1] = 0 # Domain has one guard cell where volumetric quantities are zero
            
        ## Sources
        source = 3/2 * (dom["Pd+_src"] + dom["Pe_src"]) * dom["dv"]
        sheath = abs((dom["Ee_sheath"] + dom["Ed+_sheath"]) * dom["dv"])


        ## Kinetic energy flow
        ds_m = self.ds.shift(pos = 1)   # index minux
        ds_p = self.ds.shift(pos = -1)   # index plus
        ds = self.ds
        
        def get_ke_flow(ds):
            return 0.5 * ds["Ne"] * 2*constants("mass_p") * ds["Ve"]**2 * abs(ds["Ve"]) * ds["dv"]
        
        ke_ylow = (get_ke_flow(ds_m) + get_ke_flow(ds)) / 2
        ke_yup = (get_ke_flow(ds_p) + get_ke_flow(ds)) / 2
        
        print(len(ke_ylow), len(ke_yup), len(x), ds.dims)
        
        kinetic = (ke_ylow - ke_yup).isel(pos = slice(2,-1))

        ## Total
        total = electrons + ions + source + sheath + kinetic
        if self.include_neutrals:
            total += neutrals + radiation

        style = dict(marker = "o", ms = 2, lw = 0.75)
        ax.plot(x, electrons,  c = "b", label = "Electron convection", **style)
        ax.plot(x, ions,  c = "r", label = "Ion convection", **style)
        ax.plot(x, kinetic,  c = "purple", label = "Ion KE", **style)
        ax.plot(x, source,  label = "Sources integral", **style)
        ax.plot(x, sheath,  label = "Sheath", **style)

        if self.include_neutrals:
            ax.plot(x, neutrals,  c = "grey", label = "Neutral convection", **style)
            ax.plot(x, radiation,  c = "deeppink", label = "Radiation integral", **style)

        ax.plot(x, total,  c = "black", label = "Imbalance", lw = 5, alpha = 0.2)
        ax.plot()

        ax.legend(bbox_to_anchor = (1, 1), loc = "upper left")
        ax.set_xscale("log")
        ax.set_xlabel("Distance from target [m]")
        ax.set_yscale("symlog", linthresh = 1)
        ax.set_title("Total flux balance - add Qei")
        ax.set_ylabel("Power [W]")
        ax.grid()