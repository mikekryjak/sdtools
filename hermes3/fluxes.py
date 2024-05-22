import xarray as xr
import numpy as np
from hermes3.utils import *
import pandas as pd
import warnings
warnings.simplefilter('ignore', RuntimeWarning)


"""
Power in through X inner Pin = 162418.93698190025 W  [55238.53718922597 ion 107180.39979267426 electron]
Power out through X outer Pout,x 0.0 W [0.0 ion 0.0 electron]
Power out through lower sheath: Pout,d = 41805.580325914736 W [25543.116778984073 ion 16262.463546930667 electron]
Power out through upper sheath: Pout,u = 85927.55896679485 W [40862.03993701901 ion 45065.51902977584 electron]
Domain volume: 0.44588748805918543 m^3
Thermal energy content: 49.51050515563746 J
Energy confinement time: 0.00030483209701806415 s
Net power radiation: Prad = 26396.319077657958 W [26395.075399576534 excitation 1.2436780814237998 recombination]
Net input power Pnet = Pin - Pout,x - Pout,d - Pout,u - Prad = 8289.4786115327 W
------------
Particle flux in through X inner: 4.149953576517765e+19 /s
Particle flux out through X outer: 0.0 /s
Particle flux to lower sheath: 2.0381675806634667e+21 /s
Particle flux to upper sheath: 2.2985141232757716e+21 /s
Recycling fraction: 0.990430578354994
Total ionization rate: 4.290451033846074e+21 /s
Total recombination rate: -2.5038401874305984e+17 /s
Particle content: 2.169621801152505e+18
Particle throughput time: 0.05228062823230518 s
------------
"""

def calculate_simple_particle_balance(ds):
    """
    Calculate a simple balance where things like species etc are hardcoded
    """
    m = ds.metadata

    sol = ds.hermesm.select_region("sol_edge")
    pfr = ds.hermesm.select_region("pfr_edge")
    domain = ds.hermesm.select_region("all_noguards")
    core = ds.hermesm.select_region("core_edge")

    s = {}

    # SOL
    s["sol_src_i"] = sol["pf_perp_diff_R_d+"].sum(["x", "theta"]) * -1
    s["sol_src_n"] = sol["pf_perp_diff_R_d"].sum(["x", "theta"]) * -1

    # Core
    s["core_src_i"] = core["pf_perp_diff_L_d+"].sum(["x", "theta"])
    s["core_src_n"] = core["pf_perp_diff_L_d"].sum(["x", "theta"])

    # PFR
    s["pfr_src_i"] = pfr["pf_perp_diff_L_d+"].sum(["x", "theta"])
    s["pfr_src_n"] = pfr["pf_perp_diff_L_d"].sum(["x", "theta"])

    # Wall and pump recycle
    s["pump_recycle"] = ((domain["Sd_pump"])*domain["dv"]).sum(["x", "theta"]) * m["Nnorm"] * m["Omega_ci"]
    s["wall_recycle"] = ((domain["Sd_wall_recycle"])*domain["dv"]).sum(["x", "theta"]) * m["Nnorm"] * m["Omega_ci"]
    # target_recycle = ((domain["Sd_target_recycle"])*domain["dv"]).sum(["x", "theta"])

    s["target_src_i"] = 0
    s["target_src_n"] = 0

    for name in m["targets"]:
        target_name = f"{name}_target"
        s["target_src_i"] += ds[f"pf_{target_name}_d+"].sum("x")
        s["target_src_n"] += ds[f"pf_recycle_{target_name}_d"].sum("x")
        
    targets_net = s["target_src_i"] + s["target_src_n"]

    df = pd.DataFrame(columns = ["value"])
    for name in s:
        df.loc[name, "value"] = s[name].values
        
    display_dataframe(df)
    
def calculate_simple_heat_balance(ds, verbose = True):
    """
    Simple heat balance with things hardcoded
    Currently only reflective wall cooling
    """
    if ds["t"].shape != ():
        raise Exception("Please supply a single time slice")
    places = {}

    places["inner_wall  "] = ds.hermesm.select_region("inner_sol_edge")
    places["outer_wall  "] = ds.hermesm.select_region("outer_sol_edge")
    places["pfr         "] = ds.hermesm.select_region("pfr_edge")
    hflows = {}
    
    if "Ed_wall_refl" in ds.data_vars:
        
        for place in places:
            ds_place = places[place]
            hflows[place] = ()
            hflows[place] = (ds_place["Ed_wall_refl"] * ds_place["dv"] ).sum().values * 1e-6
            
        hflows["targets     "] = (ds["Ed_target_refl"] * ds["dv"]).sum().values * 1e-6
        
        if verbose: print("Wall reflective cooling:")
        tot = 0
        for name in hflows:
            if verbose: print(f"{name}: {hflows[name]:.3f} [MW]")
            tot += hflows[name]
            
        if verbose: print(f"Total       : {tot:.3f} [MW]\n")
    else:
        if verbose: print("No wall heat fluxes found in dataset")
        
    if verbose: print("Recycling neutral energy source:")
    en_rec = (ds["Ed_target_recycle"] * ds["dv"]).sum(["x", "theta"]).values * 1e-6
    if verbose: print(f"Total       : {en_rec:.3f} [MW]")
    
    return hflows

def calculate_neutral_heat_balance(ds, mode = "default"):
    
    if ds["t"].shape != ():
        raise Exception("Please supply a single time slice")
    
    if len([x for x in ds.data_vars if "pf" in x]) == 0:
        print("WARNING: radial fluxes not found, calculating")
        ds = calculate_radial_fluxes(ds)
    
    def get_sum(param, loc = None):
    # ds = cs["base"].ds.isel(t=-1)
    
        if param in ds.data_vars:
            
            data = ds[param].hermesm.clean_guards()
            dv = ds["dv"]
            if loc != None:
                data = data.hermesm.select_region(loc)
                dv = ds.hermesm.select_region(loc)["dv"]
                
            val = (data * dv).sum(["x", "theta"]) * 1e-6
            
            if any([x in param for x in ["iz", "rec"]]):
                val *= -1
            return val
        else:
            return np.nan
        
    def get_bndry_flux(species, boundary, type = "convection"):
        if "core" in boundary:
            LR = "L"
        elif "sol" in boundary:
            LR = "R"
        else:
            raise Exception(f"{boundary} not implemented")
            
        if type == "convection":
            param = f"hf_perp_conv_{LR}_{species}"
            scale = 1e-6
        else:
            raise Exception(f"{type} not implemented")
            
        if param in ds.data_vars:
            return ds.hermesm.select_region("core_edge")[param].sum("theta").values * 1e-6
        else:
            print(f"Warning: {param} not found in dataset")
            return np.nan
        
    df = pd.DataFrame()
    df.loc["target_refl", "d"] = get_sum("Ed_target_refl")
    df.loc["inner_sol_refl", "d"] = get_sum("Ed_wall_refl", loc = "inner_sol_edge")
    df.loc["outer_sol_refl", "d"] = get_sum("Ed_wall_refl", loc = "outer_sol_edge")
    df.loc["pfr_refl", "d"] = get_sum("Ed_wall_refl", loc = "pfr_edge")
    df.loc["target_trefl", "d"] = get_sum("Edd*_target_refl")
    df.loc["wall_trefl", "d"] = get_sum("Edd*_wall_refl")
    df.loc["target_recycle", "d"] = get_sum("Ed_target_recycle") * -1
    df.loc["wall_recycle", "d"] = get_sum("Ed_wall_recycle") * -1
    df.loc["core_escape", "d"] = get_bndry_flux("d", "core_edge", type = "convection")
    df.loc["sol_escape", "d"] = get_bndry_flux("d", "sol_edge", type = "convection")
    df.loc["iz", "d"] = get_sum("Edd+_iz")
    df.loc["rec", "d"] = get_sum("Ed+_rec")
    df.loc["cx", "d"] = get_sum("Edd+_cx")
    df.loc["cxt", "d"] = get_sum("Edd+_cxt")
    df.loc["src", "d"] = get_sum("Pd_src") * 3/2

    df.loc["target_refl", "d*"] = get_sum("Ed*_target_refl")
    df.loc["inner_sol_refl", "d*"] = get_sum("Ed*_wall_refl", loc = "inner_sol_edge")
    df.loc["outer_sol_refl", "d*"] = get_sum("Ed*_wall_refl", loc = "outer_sol_edge")
    df.loc["pfr_refl", "d*"] = get_sum("Ed*_wall_refl", loc = "pfr_edge")
    df.loc["target_trefl", "d*"] = get_sum("Edd*_target_refl") * -1
    df.loc["wall_trefl", "d*"] = get_sum("Edd*_wall_refl") * -1
    df.loc["target_recycle", "d*"] = get_sum("Ed*_target_recycle")
    df.loc["wall_recycle", "d*"] = get_sum("Ed*_wall_recycle")
    df.loc["core_escape", "d*"] = get_bndry_flux("d*", "core_edge", type = "convection")
    df.loc["sol_escape", "d*"] = get_bndry_flux("d*", "sol_edge", type = "convection")
    df.loc["iz", "d*"] = get_sum("Ed*d+_iz")
    df.loc["rec", "d*"] = get_sum("Ed*d+_rec")
    df.loc["cx", "d*"] = get_sum("Ed*d+_cx")
    df.loc["cxt", "d*"] = get_sum("Ed*d+_cxt")
    
    df["total"] = df.sum(axis=1)

    df.loc["TOTAL BALANCE", :] = df[~df.index.isin(["target_trefl", "wall_trefl"])].sum(axis=0)
    
    if mode == "reflection":
        dfrefl = df.transpose()[["outer_sol_refl", "inner_sol_refl", "target_refl", "pfr_refl"]].copy()
        dfrefl = dfrefl.rename(columns = {"outer_sol_refl" : "outer_wall", "inner_sol_refl" : "inner_wall", "target_refl" : "targets", "pfr_refl" : "pfr"})
        dfrefl.loc["total", :] = dfrefl.sum(axis=0)
        dfrefl *= -1        
        return dfrefl
    
    else:
        
        return df

def calculate_heat_balance(ds, merge_targets = True):

    m = ds.metadata
    core = ds.hermesm.select_region("core_edge")
    sol = ds.hermesm.select_region("sol_edge")
    pfr = ds.hermesm.select_region("pfr_edge")
    domain = ds.hermesm.select_region("all_noguards")
    domain_volume = domain["dv"].values.sum()

    df = pd.DataFrame()
    hf = dict()

    # Radial and edge fluxes
    # The summing is clunky but there's apparently no other way in Xarray!
    net = dict()
    list_places = ["src", "core", "sol", "pfr"]
    for place in list_places:
        net[f"hf_int_{place}_net"] = 0

    for species in m["species"]:
        ds[f"hf_int_src_{species}"] = (domain[f"P{species}_src"] * domain["dv"]).sum(["x", "theta"]).squeeze() * 3/2
        ds[f"hf_int_core_{species}"] = core[f"hf_perp_tot_L_{species}"].sum("theta").squeeze()
        ds[f"hf_int_sol_{species}"] = sol[f"hf_perp_tot_R_{species}"].sum("theta").squeeze() * -1 # Positive flow at SOL leaves model
        ds[f"hf_int_pfr_{species}"] = pfr[f"hf_perp_tot_L_{species}"].sum("theta").squeeze()
        
        for place in list_places:
            net[f"hf_int_{place}_net"] += ds[f"hf_int_{place}_{species}"]
            
    for place in list_places:
        ds[f"hf_int_{place}_net"] = net[f"hf_int_{place}_net"]


    # Target fluxes
    for target in m["targets"]:
        target_name = f"{target}_target"
        net[target_name] = 0
    net["targets"] = 0

    for species in m["ion_species"]+m["neutral_species"]+["e"]:
        net[f"targets_{species}"] = 0
        for target in m["targets"]:
            target_name = f"{target}_target"
            ds[f"hf_int_{target_name}_{species}"] = ds[f"hf_{target_name}_{species}"].sum("x").squeeze()
            net[target_name] += ds[f"hf_int_{target_name}_{species}"] 
            net[f"targets_{species}"]  += ds[f"hf_int_{target_name}_{species}"] 
            
            
        ds[f"hf_int_targets_{species}"] = net[f"targets_{species}"]
            
    
    for target in m["targets"]:
        target_name = f"{target}_target"
        ds[f"hf_int_{target_name}_net"] = net[target_name]
        net["targets"] += net[target_name]
        
    ds[f"hf_int_targets_net"] = net["targets"]
        
        
    # Atomic reaction fluxes
        
    net_ex = 0
    net_rec = 0
    for ion in m["ion_species"]:
        if f"R{ion}_ex" in domain.data_vars:
            ds[f"hf_int_rad_ex_e"] = (domain[f"R{ion}_ex"] * domain["dv"]).sum(["x", "theta"]).squeeze()
            net_ex +=  ds[f"hf_int_rad_ex_e"]
            
        if f"R{ion}_rec" in domain.data_vars:
            ds[f"hf_int_rad_rec_e"] = (domain[f"R{ion}_rec"] * domain["dv"]).sum(["x", "theta"]).squeeze()       
            net_rec += ds[f"hf_int_rad_rec_e"]

    ds["hf_int_rad_ex_net"] = net_ex
    ds[f"hf_int_rad_rec_net"] = net_rec
    
    ds[f"hf_int_total_net"] = \
        ds["hf_int_src_net"] + ds["hf_int_core_net"] + ds["hf_int_sol_net"] + ds["hf_int_pfr_net"] \
            + ds["hf_int_targets_net"] + ds["hf_int_rad_ex_net"] + ds["hf_int_rad_rec_net"]
            
            
    return ds



def calculate_particle_balance(ds):
    m = ds.metadata
    
    core = ds.hermesm.select_region("core_edge")
    sol = ds.hermesm.select_region("sol_edge")
    pfr = ds.hermesm.select_region("pfr_edge")
    domain = ds.hermesm.select_region("all_noguards").squeeze()
    domain_volume = domain["dv"].values.sum()

    df = pd.DataFrame()
    pf = dict()

    # Radial and edge fluxes
    # The summing is clunky but there's apparently no other way in Xarray!
    net = dict()
    list_places = ["src", "core", "sol", "pfr"]
    for place in list_places:
        net[f"pf_int_{place}_net"] = 0

    for species in m["ion_species"] + m["neutral_species"]:
        ds[f"pf_int_src_{species}"] = (domain[f"S{species}_src"] * domain["dv"]).sum(["x", "theta"]).squeeze()
        ds[f"pf_int_core_{species}"] = core[f"pf_perp_diff_L_{species}"].sum("theta").squeeze()
        ds[f"pf_int_sol_{species}"] = sol[f"pf_perp_diff_R_{species}"].sum("theta").squeeze() * -1 # Positive is going right, which is flowing OUT
        ds[f"pf_int_pfr_{species}"] = pfr[f"pf_perp_diff_L_{species}"].sum("theta").squeeze()
    
        
        for place in list_places:
            net[f"pf_int_{place}_net"] += ds[f"pf_int_{place}_{species}"]
            
    for place in list_places:
        ds[f"pf_int_{place}_net"] = net[f"pf_int_{place}_net"]
    
    # for species in m["recycle_pair"]:
    #     ds[f"pf_int_recycle_sol_{species}"] = (domain[f"S{species}_sol_recycle"] * domain["dv"]).sum(["x", "theta"]).squeeze() 
    #     ds[f"pf_int_recycle_pfr_{species}"] = (domain[f"S{species}_pfr_recycle"] * domain["dv"]).sum(["x", "theta"]).squeeze() 
    #     ds[f"pf_int_recycle_target_{species}"] = (domain[f"S{species}_target_recycle"] * domain["dv"]).sum(["x", "theta"]).squeeze() 
        
    # Target fluxes
    for target_name in m["targets"]:
        net[target_name] = 0
    net["targets"] = 0

    for species in m["ion_species"]+m["neutral_species"]:
        net[f"targets_{species}"] = 0
        for target_name in m["targets"]:
            ds[f"pf_int_{target_name}_{species}"] = ds[f"pf_{target_name}_{species}"].sum("x").squeeze()
            net[target_name] += ds[f"pf_int_{target_name}_{species}"]
            net[f"targets_{species}"] += ds[f"pf_int_{target_name}_{species}"]
            
        ds[f"pf_int_targets_{species}"] = net[f"targets_{species}"]
            
    for target_name in m["targets"]:
        ds[f"pf_int_{target_name}_net"] = net[target_name]
        net["targets"] += net[target_name]
        
    ds[f"pf_int_targets_net"] = net["targets"]


    # Atomic reaction fluxes
    for ion in m["ion_species"]:
        
        if f"S{ion}_iz" in domain.data_vars:
            neutral = ion.split("+")[0]
            ds[f"pf_int_iz_{ion}"] = (domain[f"S{ion}_iz"] * domain["dv"]).sum(["x", "theta"]).squeeze()
            ds[f"pf_int_iz_{neutral}"] = ds[f"pf_int_iz_{ion}"] * -1
        
        if f"S{ion}_rec" in domain.data_vars:
            ds[f"pf_int_rec_{ion}"] = (domain[f"S{ion}_rec"] * domain["dv"]).sum(["x", "theta"]).squeeze()
            ds[f"pf_int_rec_{neutral}"] = ds[f"pf_int_rec_{ion}"] * -1

    # Note: no "net" ionisation or recombination since they net out to zero.
    
    ds[f"pf_int_total_net"] = \
        ds["pf_int_src_net"] + ds["pf_int_core_net"] + ds["pf_int_sol_net"] + ds["pf_int_pfr_net"] \
            + ds["pf_int_targets_net"]
            
    return ds
    
def show_heat_balance_table(ds):
    print("---------------------------------------")
    print("HEAT BALANCE")
    print("---------------------------------------")

    df = pd.DataFrame()
    m = ds.metadata

    if f"hf_int_{m['targets'][0]}" in ds.data_vars:
        merge_targets = False
    else:
        merge_targets = True

    last = ds.isel(t=-1)
    for species in m["species"]:
        df.loc["source", species] = last[f"hf_int_src_{species}"]
        df.loc["core", species] = last[f"hf_int_core_{species}"]
        df.loc["sol", species] = last[f"hf_int_sol_{species}"]
        df.loc["pfr", species] = last[f"hf_int_pfr_{species}"]
        if merge_targets is True:
            df.loc["targets", species] = last[f"hf_int_targets_{species}"]
        else:
            for target in m["targets"]:
                df.loc[target, species] = last[f"hf_int_{target}_{species}"]
        
        if "hf_int_rad_ex_e" in last.data_vars:
            df.loc["rad_ex", "e"] = last[f"hf_int_rad_ex_e"]
        if "hf_int_rad_rec_e" in last.data_vars:
            df.loc["rad_rec", "e"] = last[f"hf_int_rad_rec_e"]
        
    df["total"] = df.sum(axis=1)
    imbalance = df["total"].sum()
    imbalance_frac = imbalance / (df["total"]["source"] + df["total"]["core"])
    total_in = df["total"][df["total"]>0].sum()
    total_out = df["total"][df["total"]<0].sum()
    
    # print(f"Recycling fraction: {frec:.2%}")
    print(f"Domain volume: {ds['dv'].sum():.3e} [m3]")
    print(f"Power in: {total_in*1e-6:,.3f} [MW]")
    print(f"Power out: {total_out*1e-6:,.3f} [MW]")
    print(f"Power imbalance: {imbalance*1e-6:,.3f} [MW]")
    print(f"Power imbalance as frac of power in: {imbalance_frac:.2%}")
    print("---------------------------------------")
    print(f"Total fluxes in [MW]:")
    table = df.copy()*1e-6 # For display only

    def styler(s):
        if abs(s) < 0.01 or pd.isna(s):
            c =  "color: lightgrey"
        else:
            c =  "color: black"

        return c
        
    ts = table.style.format("{:.2f}")
    ts = ts.applymap(styler)
    display(ts)

def show_particle_balance_table(ds):
    print("---------------------------------------")
    print("PARTICLE BALANCE")
    print("---------------------------------------")

    df = pd.DataFrame()
    m = ds.metadata

    if f"pf_int_{m['targets'][0]}" in ds.data_vars:
        merge_targets = False
    else:
        merge_targets = True

    last = ds.isel(t=-1)
    for species in m["ion_species"] + m["neutral_species"]:
        df.loc["source", species] = last[f"pf_int_src_{species}"]
        df.loc["core", species] = last[f"pf_int_core_{species}"]
        df.loc["sol", species] = last[f"pf_int_sol_{species}"]
        df.loc["pfr", species] = last[f"pf_int_pfr_{species}"]
        if merge_targets is True:
            df.loc["targets", species] = last[f"pf_int_targets_{species}"]
        else:
            for target in m["targets"]:
                df.loc[target, species] = last[f"pf_int_{target}_{species}"]
        
    for species in m["ion_species"] + m["neutral_species"]:
        if f"pf_int_iz_{species}" in last.keys():
            df.loc["iz", species] = last[f"pf_int_iz_{species}"]
        if f"pf_int_rec_{species}" in last.keys():
            df.loc["rec", species] = last[f"pf_int_rec_{species}"]
        
    df["total"] = df.sum(axis=1)
    imbalance = df["total"].sum()
    imbalance_frac = imbalance / (df["total"]["source"] + df["total"]["core"])

    # print(f"Recycling fraction: {frec:.2%}")
    print(f"Domain volume: {ds['dv'].sum():.3e} [m3]")
    print(f"Particle imbalance: {imbalance:,.3e} [s-1]")
    print(f"Particle imbalance as frac of core + source: {imbalance_frac:.2%}")
    print("---------------------------------------")
    print(f"Total fluxes in [s-1]:")
    table = df.copy() # For display only

    def styler(s):
        if abs(s) < 0.01 or pd.isna(s):
            c =  "color: lightgrey"
        else:
            c =  "color: black"

        return c
        
    ts = table.style.format("{:.2e}")
    ts = ts.applymap(styler)
    display(ts)

def calculate_target_fluxes(ds):
    # ds = ds.copy() # This avoids accidentally messing up rest of dataset
    m = ds.metadata
    
    if "recycling" in ds.options.keys():
        ion = ds.options["recycling"]["species"]
        recycling = True
    else:
        ion = m["ion_species"][0] # TODO: Fix this - currently hardcoding the first ion species...
        recycling = False
    
    
    for name in m["targets"]:
        targetname = f"{name}_target"
        # targetname = name
        ds[f"hf_{targetname}_e"], ds[f"hf_{targetname}_{ion}"],  ds[f"pf_{targetname}_{ion}"] = sheath_boundary_simple(ds, ion, 
                                                                                target = name)
    
    # Get recycling fluxes
    if recycling is True:
        
        if "Sd_target_recycle" in ds.data_vars:
            print(f"Calculating target recycling:")
            for name in m["targets"]:
                print(f"{name}")
                for species in m["recycle_pair"].values():
                    targetname = f"{name}_target"
                    # targetname = name
                    target = ds.hermesm.select_region(targetname).squeeze()
                    ds[f"pf_recycle_{targetname}_{species}"] = (target[f"S{species}_target_recycle"] * target["dv"])   #[s-1]
                    ds[f"hf_recycle_{targetname}_{species}"] = (target[f"E{species}_target_recycle"] * target["dv"])   #[W]
                
                # ds[f"pf_{target}_{species}"].attrs.update(
                #     {
                #     "units" : "s-1",
                #     "standard_name": f"target particle flux on {target} ({species})",
                #     "long_name": f"target particle flux on {target} ({species})",
                #     })
                
                # pf_int_recycle_target_{species}"
                
        else:
            
            for name in m["targets"]:
                targetname = f"{name}_target"
                # targetname = name
                neutral = ds.metadata["recycle_pair"][ion]
                
                if "target_recycle_multiplier" in ds.options[ion].keys():
                    option = "target_recycle_multiplier"
                elif "recycle_multiplier" in ds.options[ion].keys():   # Old recycle fag
                    option = "recycle_multiplier"
                else:
                    raise Exception("Target recycle multiplier not found in options")
                
                ds[f"pf_recycle_{targetname}_{neutral}"] = -ds[f"pf_{targetname}_{ion}"] * ds.options[ion][option]   # TODO generalise and check
                ds[f"pf_recycle_{targetname}_{neutral}"].attrs.update(
                        {
                        "standard_name": f"target particle flux on {targetname} ({neutral})",
                        "long_name": f"target particle flux on {targetname} ({neutral})",
                        })
                
                # Assume neutral heat loss to target is zero for now
                # TODO: fix this when neutral_gamma is used
                ds[f"hf_recycle_{targetname}_{neutral}"] = xr.zeros_like(ds[f"hf_{targetname}_{ion}"])
            
        
    return ds
        

def calculate_radial_fluxes(ds, force_neumann = False, new_afn = True):   
    
    def zero_to_nan(x):
        return np.where(x==0, np.nan, x)

    m = ds.metadata
    Pnorm = m["Nnorm"] * m["Tnorm"] * constants("q_e")


    # HEAT FLUXES---------------------------------
    
    for name in m["charged_species"]:
        
        # Perpendicular heat diffusion (conduction)
        L, R  =  Div_a_Grad_perp_upwind_fast(ds, ds[f"N{name}"] * ds[f"anomalous_Chi_{name}"], constants("q_e") * ds[f"T{name}"])
        ds[f"hf_perp_diff_L_{name}"] = L 
        ds[f"hf_perp_diff_R_{name}"] = R

        # Perpendicular heat convection NOTE: 3/2 factor was initially omitted by accident
        L, R  =  Div_a_Grad_perp_upwind_fast(ds, constants("q_e") * ds[f"T{name}"] * ds[f"anomalous_D_{name}"], ds[f"N{name}"])
        ds[f"hf_perp_conv_L_{name}"] = L * 3/2
        ds[f"hf_perp_conv_R_{name}"] = R * 3/2

        # Total
        ds[f"hf_perp_tot_L_{name}"] = ds[f"hf_perp_conv_L_{name}"] + ds[f"hf_perp_diff_L_{name}"]
        ds[f"hf_perp_tot_R_{name}"] = ds[f"hf_perp_conv_R_{name}"] + ds[f"hf_perp_diff_R_{name}"]      
        
    for name in m["neutral_species"]:
        
        # Need to force guard cells to be the same as the final radial cells
        # Otherwise there are issues with extreme negative transport in the 
        # final radial cells, especially PFR and near targets.
        if force_neumann is True:
            Pd_fix = ds[f"P{name}"].copy()
            Pd_fix[{"x":1}] = Pd_fix[{"x":2}]
            Pd_fix[{"x":0}] = Pd_fix[{"x":2}]
            Pd_fix[{"x":-1}] = Pd_fix[{"x":-3}]
            Pd_fix[{"x":-2}] = Pd_fix[{"x":-3}]
            
            Nd_fix = ds[f"N{name}"].copy()
            Nd_fix[{"x":1}] = Nd_fix[{"x":2}]
            Nd_fix[{"x":0}] = Nd_fix[{"x":2}]
            Nd_fix[{"x":-1}] = Nd_fix[{"x":-3}]
            Nd_fix[{"x":-2}] = Nd_fix[{"x":-3}]
            
            Td_fix = ds[f"T{name}"].copy()
            Td_fix[{"x":1}] = Td_fix[{"x":2}]
            Td_fix[{"x":0}] = Td_fix[{"x":2}]
            Td_fix[{"x":-1}] = Td_fix[{"x":-3}]
            Td_fix[{"x":-2}] = Td_fix[{"x":-3}]
            
            Plim = Pd_fix.where(Pd_fix>0, 1e-8 * Pnorm)
            Td = Td_fix
            Nd = Nd_fix
            
        else:
            
            Plim = ds[f"P{name}"].where(ds[f"P{name}"]>0, 1e-8 * Pnorm)
            Td = ds[f"T{name}"]
            Nd = ds[f"N{name}"]
        
        # Plim = ds[f"P{name}"].where(ds[f"P{name}"]>0, 1e-8 * Pnorm) # Limit P to 1e-8 in normalised units
        
        # Perpendicular heat convection NOTE: 3/2 factor was initially omitted by accident
        L, R  =  Div_a_Grad_perp_fast(ds, ds[f"Dnn{name}"]*ds[f"P{name}"], np.log(Plim))
        
        if new_afn is True and f"particle_flux_factor_{name}" in ds.data_vars:
            corr = ds[f"particle_flux_factor_{name}"]    # Convection limited by particle diffusion factor
        else:
            corr = 1
        
        ds[f"hf_perp_conv_L_{name}"] = L * 3/2 * corr
        ds[f"hf_perp_conv_R_{name}"] = R * 3/2 * corr
        
        # Perpendicular heat conduction NOTE: this is actually energy not pressure like above, so no factor needed
        L, R  =  Div_a_Grad_perp_fast(ds, ds[f"Dnn{name}"]*Nd, constants("q_e")*Td)
        
        if new_afn is True and f"heat_flux_factor_{name}" in ds.data_vars:
            corr = ds[f"heat_flux_factor_{name}"]     # Conduction limited by heat flux factor
        else:
            corr = 1
            
        ds[f"hf_perp_diff_L_{name}"] = L * corr 
        ds[f"hf_perp_diff_R_{name}"] = R * corr
        
        # Total
        ds[f"hf_perp_tot_L_{name}"] = ds[f"hf_perp_conv_L_{name}"] + ds[f"hf_perp_diff_L_{name}"]
        ds[f"hf_perp_tot_R_{name}"] = ds[f"hf_perp_conv_R_{name}"] + ds[f"hf_perp_diff_R_{name}"]   
        

    for name in m["neutral_species"] + m["charged_species"]:
        # Add metadata
        for side in ["L", "R"]:
            for kind in ["diff", "conv", "tot"]:
                
                da = ds[f"hf_perp_{kind}_{side}_{name}"]
                da.attrs["units"] = "W"
                da.attrs["standard_name"] = f"Perpendicular heat flux ({name})"
                
                kind_long = {"diff":"diffusive", "conv":"convective", "tot":"total"}[kind]
                side_long = {"L":"LHS cell face", "R":"RHS cell face"}[side]
                
                da.attrs["long_name"] = f"Perpendicular {kind_long} heat flux on {side_long} ({name})"
                da.attrs["source"] = "xHermes"
                da.attrs["conversion"] = ""
                         
    
    # PARTICLE FLUXES---------------------------------
       
    for name in m["charged_species"]:   
        L, R  =  Div_a_Grad_perp_upwind_fast(ds, ds[f"anomalous_D_{name}"], ds[f"N{name}"])
        ds[f"pf_perp_diff_L_{name}"] = L 
        ds[f"pf_perp_diff_R_{name}"] = R
        
        
    for name in m["neutral_species"]:
        
        # Need to force guard cells to be the same as the final radial cells
        # Otherwise there are issues with extreme negative transport in the 
        # final radial cells, especially PFR and near targets.
        if force_neumann is True:
            Pd_fix = ds[f"P{name}"].copy()
            Pd_fix[{"x":1}] = Pd_fix[{"x":2}]
            Pd_fix[{"x":0}] = Pd_fix[{"x":2}]
            Pd_fix[{"x":-1}] = Pd_fix[{"x":-3}]
            Pd_fix[{"x":-2}] = Pd_fix[{"x":-3}]
            Plim = Pd_fix.where(Pd_fix>0, 1e-8 * Pnorm)
        else:
            Plim = ds[f"P{name}"].where(ds[f"P{name}"]>0, 1e-8 * Pnorm)
        
        L, R  =  Div_a_Grad_perp_fast(ds, ds[f"Dnn{name}"]*ds[f"N{name}"], np.log(Plim))
        corr = ds["particle_flux_factor_d"] if "particle_flux_factor_d" in ds.data_vars else 1
        ds[f"pf_perp_diff_L_{name}"] = L * corr 
        ds[f"pf_perp_diff_R_{name}"] = R * corr
        
        
        
    for name in m["neutral_species"] + m["charged_species"]:
        for side in ["L", "R"]:
                for kind in ["diff"]:
                    da = ds[f"pf_perp_{kind}_{side}_{name}"]
                    da.attrs["units"] = "s-1"
                    da.attrs["standard_name"] = f"Perpendicular particle flux ({name})"
                
                    kind_long = {"diff":"diffusive", "conv":"convective", "tot":"total"}[kind]
                    side_long = {"L":"LHS cell face", "R":"RHS cell face"}[side]
                    
                    da.attrs["long_name"] = f"Perpendicular diffusive particle flux on {side_long} ({name})"
                    da.attrs["source"] = "xHermes"
                    da.attrs["conversion"] = ""
                
                
        
    # for name in m["neutral_species"]:
        
        
    return ds
    
def reverse_pfr_fluxes(ds):
    m = ds.metadata
    regions = [
    dict(x=slice(0, m["ixseps1"]), theta=np.r_[slice(None, m["j1_1g"] + 1), slice(m["j2_2g"] + 1, m["nyg"])]),
    dict(x=slice(0, m["ixseps1"]), theta=slice(m["j2_1g"] + 1, m["j1_2g"] + 1)),
    ]
        
    fluxes_to_correct = [x for x in ds.data_vars if "f_perp" in x]
    fluxes_to_correct += [x for x in ds.data_vars if "Flow_" in x]

    for variable in fluxes_to_correct:
        for selection in regions:
            ds[variable][selection] = ds[variable][selection] * -1
            
    return ds

def sheath_boundary_simple(bd, species, target,
                           sheath_ion_polytropic=1.0, include_convective=True):
    """    
    Calculate the electron and ion heat flux at the sheath, using the formula used in the
    sheath_boundary_simple component for a user requested ion species.
    Takes the first domain cell and guard cell indices as the input. This is how
    you select which divertor you want to look at.
    
    y = final domain cell index
    y2 = second to final domain cell index
    yg = first guard cell index
    
    BOUT++ notation:
    y = final domain cell index
    ym = previous cell along poloidal
    yp = next cell along poloidal

    # Returns

    flux_down, flux_up

    With units of Watts, i.e the power flowing out of each cell

    Slices at lower Y and upper Y respectively, giving heat conduction through sheath.
    Note: These do *not* include the convective heat flux, since that would usually
    be calculated in the pressure evolution (evolve_pressure component).
    """
    
    m = bd.metadata
    bd = bd.isel(x = slice(m["MYG"], -m["MYG"])) # No radial guard cells allowed
    target_indices = dict()
    
    # TODO: Integrate with select_region
    if m["keep_yboundaries"] == 0:
        target_indices["inner_lower"] = dict(y = 0, y2 = 1, yg = None)
        target_indices["outer_lower"] = dict(y = -1, y2 = -2, yg = None)
        target_indices["inner_upper"] = dict(y = m["ny_inner"]-1, y2 = m["ny_inner"]-2, yg = None)
        target_indices["outer_upper"] = dict(y = m["ny_inner"]+1, y2 = m["ny_inner"]+2, yg = None)
    else:
        target_indices["inner_lower"] = dict(y = m["MYG"], y2 = m["MYG"]+1, yg = m["MYG"]-1)
        target_indices["outer_lower"] = dict(y = -m["MYG"]-1, y2 = -m["MYG"]-2, yg = -m["MYG"])
        target_indices["inner_upper"] = dict(y = m["ny_inner"] + m["MYG"]-1, y2 = m["ny_inner"], yg = m["ny_inner"] + m["MYG"])
        target_indices["outer_upper"] = dict(y = m["ny_inner"] + m["MYG"]*3, y2 = m["ny_inner"]+m["MYG"]*3+1, yg = m["ny_inner"] + m["MYG"]*3 - 1)
    
    idx = target_indices[target]
    y = idx["y"] # Final domain cell 
    y2 = idx["y2"] # Second to final domain cell
    yg = idx["yg"] # Inner guard cell
    
    reproduce_error = False  # There was a mistake in the original version of this code.
    
    if "sheath_boundary_simple" in bd.options.keys():
        if "gamma_e" in bd.options["sheath_boundary_simple"].keys():
            gamma_e = bd.options["sheath_boundary_simple"]["gamma_e"]
        else:
            gamma_e = 3.5   # Hermes-3 default
        
        if "gamma_e" in bd.options["sheath_boundary_simple"].keys():
            gamma_i = bd.options["sheath_boundary_simple"]["gamma_i"]
        else:
            gamma_i = 3.5   # Hermes-3 default
    else:
        print("Warning: sheath_boundary_simple not found in settings. Assuming it is enabled")
        gamma_i = 3.5
        gamma_e = 3.5
        
    Zi = bd.options[f"{species}"]["charge"]
    AA = bd.options[f"{species}"]["AA"]

    Ne = bd["Ne"]
    Te = bd["Te"]
    Ti = bd[f"T{species}"]
    Vi = bd[f"V{species}"]
    
    if not include_convective:
        gamma_e = gamma_e - 2.5
        gamma_i = gamma_i - 3.5
    
    J = bd['J']
    dx = bd['dx']
    dy = bd['dy']
    dz = bd['dz']
    g_22 = bd['g_22']


    if m["keep_yboundaries"] == 1:
        Ne_g = Ne.isel(theta=yg)
        Te_g = Te.isel(theta=yg)
        Ti_g = Ti.isel(theta=yg)
    else:
        print("Warning: Y guard cells not present, extrapolating!")
        yg = y
        Ne_g = Ne.isel(theta=y)**2 / Ne.isel(theta=y2)
        Te_g = Te.isel(theta=y)**2 / Te.isel(theta=y2)
        Ti_g = Ti.isel(theta=y)**2 / Ti.isel(theta=y2)

    # Guard replace
    nesheath = 0.5 * (Ne.isel(theta=y) + Ne_g)
    tesheath = 0.5 * (Te.isel(theta=y) + Te_g)
    tisheath = 0.5 * (Ti.isel(theta=y) + Ti_g)

    qe = 1.602e-19 # Elementary charge [C]
    mp = 1.67e-27 # Proton mass [kg]
    me = 9.11e-31 # Electron mass [kg]
    
    # Ion flow speed
    C_i = np.sqrt((sheath_ion_polytropic * qe * tisheath + Zi * qe * tesheath) / (AA * mp))

    vesheath = C_i  # Assuming no current
    
    # Allow to go supersonic based on final cell value
    # Note in this function V is always positive, in Hermes-3 there is a vertical direction dependency
    
    vesheath = np.where(vesheath > abs(Vi.isel(theta=y)), vesheath, abs(Vi.isel(theta=y)))
    
    # Parallel heat flux in W/m^2.
    # Note: Corrected for 5/2Pe convective thermal flux, and small electron kinetic energy flux
    # so gamma_e is the total *energy* flux coefficient.
    if reproduce_error is True:
        if y == -1:
            q_e = (gamma_e * qe * tesheath - 0.5 * me * vesheath**2) * nesheath * vesheath
        else:
            q_e = ((gamma_e - 2.5) * qe * tesheath - 0.5 * me * vesheath**2) * nesheath * vesheath
    else:
        q_e = (gamma_e * qe * tesheath - 0.5 * me * vesheath**2) * nesheath * vesheath
    q_i = gamma_i * qe * tisheath * nesheath * vesheath

    # Multiply by cell area to get power
    hf_e = q_e * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yg)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yg)))
    hf_i = q_i * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yg)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yg)))

    # Ion flux [s-1]
    pf_i = nesheath * vesheath * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yg)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yg)))
    
    # Diagnostic prints
    # if target == "outer_lower":
    #     factor = (J.isel(theta=y) + J.isel(theta=yg)) #/ (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yg)))
        
    # print(nesheath.isel(t=-1).mean().values)
    # print(vesheath.mean())
        # print(dx.mean().values)
        # print(g_22.isel(theta=y).mean().values)
        # print(g_22.isel(theta=yg).mean().values)
        
        # print(J.isel(theta=y).mean().values)
        # print(J.isel(theta=yg).mean().values)
        
        # print(dx.isel(theta=y).mean().values)
        # print(dx.isel(theta=yg).mean().values)
        
        # print(dz.mean().values)
        # print(factor.mean().values)

        # print(J.isel(theta=y).mean().values)
        # print(f"nesheath: {nesheath.isel(t=-1).sum():.3e}")
        # print(f"vesheath: {vesheath.isel(t=-1).sum():.3e}")
        # print(f"dx: {dx.isel(t=-1).sum():.3e}")
        # print(f"dz: {dz.isel(t=-1).sum():.3e}")
    
    # Negative means leaving the model
    hf_e *= -1
    hf_i *= -1
    pf_i *= -1
    
    hf_i.attrs["units"] = "W"
    hf_e.attrs["units"] = "W"
    pf_i.attrs["units"] = "s-1"
    
    hf_i.attrs["standard_name"] = f"target heat flux ({species})"
    hf_e.attrs["standard_name"] = f"target heat flux ({species})"
    pf_i.attrs["standard_name"] = f"target particle flux ({species})"
    
    hf_i.attrs["long_name"] = f"target heat flux ({species})"
    hf_e.attrs["long_name"] = f"target heat flux ({species})"
    pf_i.attrs["long_name"] = f"target particle flux ({species})"

    return hf_e.squeeze(), hf_i.squeeze(),  pf_i.squeeze(), 
    
def sheath_boundary_simple_original(bd, gamma_e, gamma_i, Ne, Te, Ti, Zi=1, AA=1, sheath_ion_polytropic=1.0,
                           include_convective=True):
    """
    Calculate the electron and ion heat flux at the sheath, using the formula used in the
    sheath_boundary_simple component, assuming a single ion species
    with charge Zi (hydrogen=1) and atomic mass AA (hydrogen=1)

    # Returns

    flux_down, flux_up

    With units of Watts, i.e the power flowing out of each cell

    Slices at lower Y and upper Y respectively, giving heat conduction through sheath.
    Note: These do *not* include the convective heat flux, since that would usually
    be calculated in the pressure evolution (evolve_pressure component).
    """

    if not include_convective:
        gamma_e = gamma_e - 2.5
        gamma_i = gamma_i - 3.5
    
    J = bd['J']
    dx = bd['dx']
    dy = bd['dy']
    dz = bd['dz']
    g_22 = bd['g_22']

    # Lower y
    if bd.metadata["keep_yboundaries"]:
        y = 2 # First point in the domain
        ym = y - 1
        Ne_m = Ne.isel(theta=ym)
        Te_m = Te.isel(theta=ym)
        Ti_m = Ti.isel(theta=ym)
    else:
        y = 0
        ym = y # Same metric tensor component in boundary cells as in domain
        yp = y + 1 # For extrapolating boundary
        
        Ne_m = Ne.isel(theta=y)**2 / Ne.isel(theta=yp)
        Te_m = Te.isel(theta=y)**2 / Te.isel(theta=yp)
        Ti_m = Ti.isel(theta=y)**2 / Ti.isel(theta=yp)

    nesheath = 0.5 * (Ne.isel(theta=y) + Ne_m)
    tesheath = 0.5 * (Te.isel(theta=y) + Te_m)
    tisheath = 0.5 * (Ti.isel(theta=y) + Ti_m)

    qe = 1.602e-19 # Elementary charge [C]
    mp = 1.67e-27 # Proton mass [kg]
    me = 9.11e-31 # Electron mass [kg]
    
    # Ion flow speed
    C_i = np.sqrt((sheath_ion_polytropic * qe * tisheath + Zi * qe * tesheath) / (AA * mp))

    vesheath = C_i  # Assuming no current

    # Parallel heat flux in W/m^2.
    # Note: Corrected for 5/2Pe convective thermal flux, and small electron kinetic energy flux
    # so gamma_e is the total *energy* flux coefficient.
    q_e = ((gamma_e - 2.5) * qe * tesheath - 0.5 * me * vesheath**2) * nesheath * vesheath
    q_i = gamma_i * qe * tisheath * nesheath * vesheath

    # Multiply by cell area to get power
    flux_down_e = q_e * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=ym)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=ym)))
    flux_down_i = q_i * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=ym)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=ym)))

    ions_down = nesheath * vesheath * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=ym)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=ym)))
    
    # Repeat for upper Y boundary
    if bd.metadata["keep_yboundaries"] == 1:
        y = -3 # First point in the domain
        yp = y + 1
        Ne_p = Ne.isel(theta=yp)
        Te_p = Te.isel(theta=yp)
        Ti_p = Ti.isel(theta=yp)
    else:
        y = -1
        yp = y # Same metric tensor component in boundary cells as in domain
        ym = y - 1 # For extrapolating boundary
        
        Ne_p = Ne.isel(theta=y)**2 / Ne.isel(theta=ym)
        Te_p = Te.isel(theta=y)**2 / Te.isel(theta=ym)
        Ti_p = Ti.isel(theta=y)**2 / Ti.isel(theta=ym)

    nesheath = 0.5 * (Ne.isel(theta=y) + Ne_p)
    tesheath = 0.5 * (Te.isel(theta=y) + Te_p)
    tisheath = 0.5 * (Ti.isel(theta=y) + Ti_p)
    C_i = np.sqrt((sheath_ion_polytropic * qe * tisheath + Zi * qe * tesheath) / (AA * mp))
    vesheath = C_i
    
    q_e = (gamma_e * qe * tesheath - 0.5 * me * vesheath**2) * nesheath * vesheath
    q_i = gamma_i * qe * tisheath * nesheath * vesheath

    flux_up_e = q_e * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yp)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yp)))
    flux_up_i = q_i * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yp)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yp)))

    ions_up = nesheath * vesheath * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yp)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yp)))

    return flux_down_e, flux_up_e, flux_down_i, flux_up_i, ions_down, ions_up

def Div_a_Grad_perp_upwind_fast(ds, a, f):
    """
    AUTHOR: M KRYJAK 15/03/2023
    Rewrite of Div_a_Grad_perp_upwind but using fast Xarray functions
    Runs very fast, but cannot be easily verified against the code
    Full comments available in the slower version
    
    Parameters
    ----------
    bd : xarray.Dataset
        Dataset with the simulation results for geometrical info
    a : np.array
        First term in the derivative equation, e.g. chi * N
    f : np.array
        Second term in the derivative equation, e.g. q_e * T
    
    Returns
    ----------
    Tuple of two quantities:
    (F_L, F_R)
    These are the flow into the cell from the left (x-1),
    and the flow out of the cell to the right (x+1). 
    NOTE: *NOT* the flux; these are already integrated over cell boundaries

    Example
    ----------
    F_L, F_R = Div_a_Grad_perp_upwind(ds, chi * N, e * T)
    - chi is the heat diffusion in m^2/2
    - N is density in m^-3
    - T is temperature in eV
    Then F_L and F_R would have units of Watts,
    The time-derivative of the pressure in that cell would be:
    d/dt (3/2 P) = (F_L - F_R) / V
    where V = dx * dy * dz * J is the volume of the cell
    
    """
    
    J = ds["J"] # Jacobian
    g11 = ds["g11"]
    dx = ds["dx"]
    dy = ds["dy"]
    dz = ds["dz"]
    
    # f = f[i], fp = f[i+1]
    ap = a.shift(x=-1)
    fp = f.shift(x=-1)
    Jp = J.shift(x=-1)
    g11p = g11.shift(x=-1)
    dxp = dx.shift(x=-1)
    

    gradient = (fp - f) * (J*g11 + Jp*g11p) / (dx + dxp)
    
    # Upwind scheme: if gradient positive, yield a[i+1]. If gradient negative, yield a[i]
    # xr.where: where true, yield a, else yield something else
    flux = -gradient * a.where(gradient<0, ap)
    flux *= dy * dz
    
    F_R = flux
    F_L = flux.shift(x=1)  # the shift of 1 index to get F_L because the left flux at cell X is the same as the right flux at cell X-1.
    
    return F_L, F_R



def Div_a_Grad_perp_fast_broken(ds, a, f):
    """
    MK: Tried to reproduce what's in the code.
    Gives nonsense answers, not sure why.
    ----------
    -
    
    """
    result = xr.zeros_like(a)
    J = ds["J"]
    g11 = ds["g11"]
    dx = ds["dx"]
    dy = ds["dy"]
    dz = ds["dz"]
    
    # f = f[i], fp = f[i+1]
    ap = a.shift(x=-1)
    fp = f.shift(x=-1)
    Jp = J.shift(x=-1)
    g11p = g11.shift(x=-1)
    dxp = dx.shift(x=-1)
    resultp = result.shift(x=-1)
    
    # fout = 0.5 * (a + ap) * (J*g11 + Jp*g11p) * (fp - f)/(dx + dxp)
    fout = (fp - f) * (J*g11 + Jp*g11p) / (dx + dxp) * 0.5 * (a + ap)
    result -= fout #/ (dx*J) # No need to divide by length since we want flux
    resultp += fout #/ (dxp*Jp)
    
    result *= dy*dz
    
    
    
    F_R = result
    F_L = result.shift(x=1)  # the shift of 1 index to get F_L because the left flux at cell X is the same as the right flux at cell X-1.
    
    return F_L, F_R

def Div_a_Grad_perp_fast(ds, a, f):
    """
    AUTHOR: M KRYJAK 30/03/2023
    Reproduction of Div_a_Grad_perp from fv_ops.cxx in Hermes-3
    This is used in neutral parallel transport operators but not in 
    plasma anomalous diffusion
    
    Parameters
    ----------
    bd : xarray.Dataset
        Dataset with the simulation results for geometrical info
    a : np.array
        First term in the derivative equation, e.g. chi * N
    f : np.array
        Second term in the derivative equation, e.g. q_e * T
    
    Returns
    ----------
    Tuple of two quantities:
    (F_L, F_R)
    These are the flow into the cell from the left (x-1),
    and the flow out of the cell to the right (x+1). 
    NOTE: *NOT* the flux; these are already integrated over cell boundaries

    Example
    ----------
    -
    
    """
    result = xr.zeros_like(a)
    J = ds["J"]
    g11 = ds["g11"]
    dx = ds["dx"]
    dy = ds["dy"]
    dz = ds["dz"]
    
    # f = f[i], fp = f[i+1]
    ap = a.shift(x=-1)
    fp = f.shift(x=-1)
    Jp = J.shift(x=-1)
    g11p = g11.shift(x=-1)
    dxp = dx.shift(x=-1)
    resultp = result.shift(x=-1)
    
    gradient = (fp - f) * (J*g11 + Jp*g11p) / (dx + dxp)
    
    # Upwind scheme: if gradient positive, yield a[i+1]. If gradient negative, yield a[i]
    # xr.where: where true, yield a, else yield something else
    flux = -gradient * 0.5 * (a + ap)
    flux *= dy * dz
    
    F_R = flux
    F_L = flux.shift(x=1)  # the shift of 1 index to get F_L because the left flux at cell X is the same as the right flux at cell X-1.
    
    return F_L, F_R


def Div_a_Grad_perp_upwind(bd, a, f):
    """
    AUTHOR: B DUDSON 2023
    Reproduction of the same function within Hermes-3 
    Used to calculate perpendicular fluxes
    Runs very slow, but can be easily verified against the code
    
    # Returns

    Tuple of two quantities:
    (F_L, F_R)

    These are the flow into the cell from the left (x-1),
    and the flow out of the cell to the right (x+1). 

    Note: *NOT* the flux; these are already integrated over cell boundaries

    # Example

    F_L, F_R = Div_a_Grad_perp_upwind(chi * N, e * T)

    - chi is the heat diffusion in m^2/2
    - N is density in m^-3
    - T is temperature in eV

    Then F_L and F_R would have units of Watts,

    The time-derivative of the pressure in that cell would be:

    d/dt (3/2 P) = (F_L - F_R) / V

    where V = dx * dy * dz * J is the volume of the cell
    
    """

    J = bd["J"] # Jacobian
    g11 = bd["g11"]
    dx = bd["dx"]
    dy = bd["dy"]
    dz = bd["dz"]

    F_R = xr.zeros_like(f) # Flux to the right
    F_L = xr.zeros_like(f) # Flux from the left

    for x in bd.x[:-1]:
        xp = x + 1  # The next X cell
        # Note: Order of array operations matters for shape of the result
        gradient = (f.isel(x=xp) - f.isel(x=x)) * (J.isel(x=x) * g11.isel(x=x) + J.isel(x=xp) * g11.isel(x=xp))  / (dx.isel(x=x) + dx.isel(x=xp))

        flux = -gradient * 0.5*(a.isel(x=x) + a.isel(x=xp))
        
        # if gradient > 0:
        #     # Flow from x+1 to x
        #     flux = -gradient * a.isel(x=xp)  # Note: Negative flux = flow into this cell from right
        # else:
        #     # Flow from x to x+1
        #     flux = -gradient * a.isel(x=x)  # Positive flux => Flow from this cell to the right

        # Need to multiply by dy * dz because these are assumed constant in X in the calculation
        # of flux and cell volume.
        flux *= dy.isel(x=x) * dz.isel(x=x)

        F_R[dict(x=x)] = flux
        F_L[dict(x=xp)] = flux

    return F_L, F_R