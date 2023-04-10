
  
def slicer(self, name):
    """
    DOUBLE NULL ONLY
    Pass this touple to a field of any parameter spanning the grid
    to select points of the appropriate region.
    Each slice is a tuple: (x slice, y slice)
    Use it as: selected_array = array[slice] where slice = (x selection, y selection) = output from this method.
    """

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
        
    def custom_sol_ring(i, region):
        """
        Creates custom SOL ring slice beyond the separatrix.
        args[0] = i = index of SOL ring (0 is separatrix, 1 is first SOL ring)
        args[1] = region = all, inner, inner_lower, inner_upper, outer, outer_lower, outer_upper
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


def plot_slice(self, slicer, dpi = 100):
    """
    Indicates region of cells in X, Y and R, Z space for implementing Hermes-3 sources
    You must provide a slice() object for the X and Y dimensions which is a tuple in the form (X,Y)
    X is the radial coordinate (excl guards) and Y the poloidal coordinate (incl guards)
    WARNING: only developed for a connected double null. Someone can adapt this to a single null or DDN.
    """
    
    meta = self.ds.metadata
    xslice = slicer[0]
    yslice = slicer[1]

    # Region boundaries
    ny = meta["ny"]     # Total ny cells (incl guard cells)
    nx = meta["nx"]     # Total nx cells (excl guard cells)
    Rxy = self.ds["R"].values    # R coordinate array
    Zxy = self.ds["Z"].values    # Z coordinate array
    MYG = meta["MYG"]

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

## Below was replaced with a more Xarray version on 04/04/2023
def calculate_heat_balance(ds, show = True, merge_targets = True):
    print("---------------------------------------")
    print("HEAT BALANCE")
    print("---------------------------------------")

    m = ds.metadata
    core = ds.hermesm.select_region("core_edge")
    sol = ds.hermesm.select_region("sol_edge")
    pfr = ds.hermesm.select_region("pfr_edge")
    domain = ds.hermesm.select_region("all_noguards")
    domain_volume = domain["dv"].values.sum()

    df = pd.DataFrame()
    hf = dict()

    # Radial and edge fluxes
    for species in m["species"]:
        hf[species] = dict()
        hf[species]["source"] = (domain[f"P{species}_src"] * domain["dv"]).sum(["x", "theta"]).squeeze() * 3/2
        hf[species]["core"] = core[f"hf_perp_tot_L_{species}"].sum("theta").squeeze()
        hf[species]["sol"] = sol[f"hf_perp_tot_R_{species}"].sum("theta").squeeze()
        hf[species]["pfr"] = pfr[f"hf_perp_tot_L_{species}"].sum("theta").squeeze()


    # Target fluxes
    for species in m["charged_species"]:
        
        if merge_targets is True:
            hf[species]["targets"] = 0
            for target_name in m["targets"]:
                hf[species]["targets"] += ds[f"hf_{target_name}_{species}"].sum("x").squeeze()
        else:
            for target_name in m["targets"]:
                hf[species][target_name] = ds[f"hf_{target_name}_{species}"].sum("x").squeeze()
            

    # Atomic reaction fluxes
    hf["e"]["rad_ex"] = (domain["Rd+_ex"] * domain["dv"]).sum(["x", "theta"]).squeeze()
    hf["e"]["rad_rec"] = (domain["Rd+_rec"] * domain["dv"]).sum(["x", "theta"]).squeeze()

    # Assemble final timesteps
    hf_last = dict()

    df = pd.DataFrame()
    for species in hf.keys():
        hf_last[species] = dict()
        for loc in hf[species].keys():
            hf_last[species][loc] = hf[species][loc].values[-1] * 1e-6    # EVERYTHING IN MW
            
    df = pd.DataFrame.from_dict(hf_last)
    df["total"] = df.sum(axis=1)

    totals = pd.DataFrame(columns = df.columns)
    totals.loc["total"] = df.sum(axis=0)


    totals.loc["total(frac)"] = totals.loc["total"] / df.loc[["source", "core"], :].sum(axis=0)

    if merge_targets is True:
        target_total = hf_last["d+"]["targets"] + hf_last["e"]["targets"]
    else:
        target_total = np.sum([hf_last["d+"][x]+hf_last["e"][x] for x in m["targets"]])

    imbalance = df["total"].sum()
    imbalance_frac = imbalance / (df["total"]["source"] + df["total"]["core"])
    
    if show is True:
        # print(f"Recycling fraction: {frec:.2%}")
        print(f"Domain volume: {domain['dv'].sum():.3e} [m3]")
        print(f"Power imbalance: {imbalance*1e6:,.0f} [W]")
        print(f"Power imbalance as frac of core + source: {imbalance_frac:.2%}")
        table = pd.concat([df,totals])
        
        def styler(s):
            if abs(s) < 0.01 or pd.isna(s):
                c =  "color: lightgrey"
            else:
                c =  "color: black"

            
            return c
            
        ts = table.style.format("{:.2f}")
        ts = ts.applymap(styler)
        display(ts)
        
## Below was replaced with a more Xarray version on 04/04/2023
def calculate_particle_balance(ds, show = True, merge_targets = True):
    """
    NOTE: DOES NOT PRECISELY REPLICATE PAPER BUT IS VERY CLOSE. INVESTIGATE
    """
    
    print("---------------------------------------")
    print("PARTICLE BALANCE")
    print("---------------------------------------")
    
    m = ds.metadata
    
    core = ds.hermesm.select_region("core_edge")
    sol = ds.hermesm.select_region("sol_edge")
    pfr = ds.hermesm.select_region("pfr_edge")
    domain = ds.hermesm.select_region("all_noguards").squeeze()
    domain_volume = domain["dv"].values.sum()

    heavy_species = m["ion_species"] + m["neutral_species"]

    df = pd.DataFrame()

    # for species in heavy_species:
    #     df.loc["core", species] = 

    pf = dict()
    hf = dict()

    # pf_total = dict()

    for species in heavy_species:
        pf[species] = dict()
        # Radial fluxes
        pf[species]["source"] = (domain[f"S{species}_src"] * domain["dv"]).sum(["x", "theta"]).squeeze()
        pf[species]["core"] = core[f"pf_perp_diff_L_{species}"].sum("theta").squeeze()
        pf[species]["sol"] = sol[f"pf_perp_diff_R_{species}"].sum("theta").squeeze()
        pf[species]["pfr"] = pfr[f"pf_perp_diff_L_{species}"].sum("theta").squeeze()

        # Target fluxes
        if merge_targets is True:
            pf[species]["targets"] = 0
            for target_name in m["targets"]:
                pf[species]["targets"] += ds[f"pf_{target_name}_{species}"].sum("x").squeeze()
        else:
            for target_name in m["targets"]:
                pf[species][target_name] = ds[f"pf_{target_name}_{species}"].sum("x").squeeze()

    for species in m["ion_species"]:      
        # Atomic reaction fluxes
        pf[species]["iz"] = (domain["Sd+_iz"] * domain["dv"]).sum(["x", "theta"]).squeeze()
        pf[species]["rec"] = (domain["Sd+_rec"] * domain["dv"]).sum(["x", "theta"]).squeeze()
        
        neutral = species.replace("+","")
        
        # Neutral partner has opposite fluxes
        pf[neutral]["iz"] = pf[species]["iz"] * -1
        pf[neutral]["rec"] = pf[species]["rec"] * -1
        
    pf_last = dict()

    df = pd.DataFrame()
    for species in pf.keys():
        pf_last[species] = dict()
        for loc in pf[species].keys():
            pf_last[species][loc] = pf[species][loc].values[-1]
            
    df = pd.DataFrame.from_dict(pf_last)
    df["total"] = df["d+"] + df["d"] 

    totals = pd.DataFrame(columns = df.columns)
    totals.loc["total"] = df.sum(axis=0)
    totals.loc["total(frac)"] = totals.loc["total"] / df.loc[["source", "core"], :].sum(axis=0)

    if merge_targets is True:
        target_total = pf[species]["targets"].isel(t=-1)
    else:
        target_total = np.sum([pf_last["d+"][x] for x in m["targets"]])

    frec = 1 - (pf_last["d+"]["core"] + pf_last["d+"]["source"]) / abs(target_total)

    if show is True:
        table = pd.concat([df,totals])
        
        def styler(s):
            if abs(s) < 0.01 or pd.isna(s):
                c =  "color: lightgrey"
            else:
                c =  "color: black"

            return c

        print(f"Recycling fraction: {frec:.2%}")
        print(f"Domain volume: {domain['dv'].sum():.3e}")
        
        ts = table.style.format("{:.3e}")
        ts = ts.applymap(styler)
        display(ts)
    
    else:
        return pf