import regex as re
import numpy as np
import xarray as xr

class Balance1D():
    
    def __init__(self, ds):
        print("***Warning, species choice currently hardcoded\n")
        print("***Warning, you must have guard cells loaded in\n")
        
        self.ds = ds
        self.dom = ds.isel(pos = slice(2,-2))   # Domain
        self.terms_extracted = False
        self.tallies_extracted = False
        
        # Check for sheath diagnostic variables
        sheath_flux_candidates = [var for var in ds.data_vars if re.search("S.*\+_sheath", var)]
        if len(sheath_flux_candidates) == 0:
            raise ValueError("No sheath particle flux diagnostics found, check if Hermes-3 is saving it")
        
        # Check if more than one time slice
        if "t" in ds.dims:
            self.time = True
        else:
            self.time = False
        
        
    def get_terms(self):
        """
        Extract individual balance terms based on available diagnostics
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
                print(f"Warning! {param} not found, results may be incorrect")
       
                
        ### Heat balance
        hbal = {}
        hbalsum = {}
        for param in ["Pd+_src", "Pe_src", "Pd_src", "Rd+_ex", "Rd+_rec", "Rar", "Ed_target_recycle", "Ee_sheath", "Ed+_sheath"]:
            if param in ds:
                hbal[param] = integral(dom[param])*1e-6   # MW

                if param.startswith("P"):
                    hbal[param] *= 3/2    # Convert from pressure to energy
            else:
                print(f"Warning! {param} not found, results may be incorrect")
        
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
        pbalsum["recycle_frac"] = abs(pbal["Sd_target_recycle"] / pbal["Sd+_sheath"])
        pbalsum["net_iz"] = pbal["Sd+_iz"] + pbal["Sd+_rec"]
        pbalsum["sources_and_net_iz"] = pbalsum["sources_sum"] + pbalsum["net_iz"]
        pbalsum["sheath_sum"] = pbal["Sd+_sheath"] + pbal["Sd_target_recycle"]
        
        pbalerror = {}
        pbalerror["imbalance"] = pbalsum["sources_and_net_iz"] + pbalsum["sheath_sum"]
        
        ## Heat balance
        hbal = self.hbal
        hbalsum = {}
        hbalsum["sources_sum"] = hbal["Pd+_src"] + hbal["Pe_src"] + hbal["Pd_src"] 
        hbalsum["R_hydr_sum"] = hbal["Rd+_ex"] + hbal["Rd+_rec"]
        hbalsum["R_imp_sum"] = hbal["Rar"] if "Rar" in ds else np.zeros_like(hbal["Pd+_src"])
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
        
    def print_balances(self):
        pbal = self.pbal
        hbal = self.hbal
        pbalsum = self.pbalsum
        hbalsum = self.hbalsum
        pbalerror = self.pbalerror
        hbalerror = self.hbalerror
        

        
        ### Print output for particle balance
        print("\nParticle flows [s^-1]")
        print("Check recycle fraction against expectations")
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
        error = (pbalsum["sources_and_net_iz"] + pbalsum["sheath_sum"])
        print(f"Imbalance: Src + IZ+REC + Sheath: {error[-1]:.3e}")

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