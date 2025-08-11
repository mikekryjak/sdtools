from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
from hermes3.plotting import *
from hermes3.front_tracking import *
import numpy as np
import scipy

def _get_poloidal_range(ds, region):
    """
    Returns poloidal region start and end indices in a tuple.
    Each region is a quarter of a CDN topology corresponding to a 
    divertor. The regions are all in a positive index order.
    The regions extend past the midplane by one point.
    """
    m = ds.metadata
    
    if m["topology"] == "connected-double-null":

        ## This comes from _select_custom_sol_ring
        if region == "inner_lower":
            start = m["MYG"]
            end = m["imp_a"] + 1
        elif region == "inner_upper":
            start = m["imp_b"]
            end = m["ny_inner"] + m["MYG"]
        elif region == "outer_lower":
            start = m["omp_a"]
            end = m["nyg"] - m["MYG"]
        elif region == "outer_upper":
            start = m["ny_inner"] + m["MYG"]*3
            end = m["omp_b"]+1
        
        else:
            raise ValueError(f"Unknown region {region} for poloidal range")
        
    else:
        raise ValueError(f"Topology {m['topology']} not yet supported")
    
    return (start,end)

def get_1d_radial_data(ds, 
                       params, 
                       region = None, 
                       poloidal_index = None, 
                       guards = False,
                       sol = True,
                       core = True
                       ):
    """
    Return a Pandas Dataframe with a radial slice of data in the given region
    The dataframe contains a radial distance normalised to the separatrix.
    Guards are excluded. if OMP or IMP is selected, it will interpolate onto Z = 0.
    Regions = omp, imp, inner_lower, inner_upper, outer_lower, outer_upper.
    alternatively, you can specify a poloidal index to get data at that index.
    """
        
    df = pd.DataFrame()
    m = ds.metadata
    
    if guards:
        xslice = slice(None, None)
    else:
        xslice = slice(m["MXG"], -m["MXG"])
        
    if region is None and poloidal_index is None:
        raise Exception("Please specify region or poloidal_index")
    
    # Catch alternative names for some regions
    translate_dict = {
        "outer_midplane" : "omp",
        "inner_midplane" : "imp"
    }
    if region in translate_dict:
        region = translate_dict[region]
    
    ## Choose region and get data
    # Interpolate to Z = 0 for IMP or OMP
    if region == "omp" or region == "imp":
       
        if region == "omp":
            reg = ds.isel(x = xslice, theta = slice(m["omp_a"] - 2, m["omp_b"] + 2))
        else:
            reg = ds.isel(x = xslice, theta = slice(m["imp_a"] - 2, m["imp_b"] + 2))
        
        # For every parameter, collect the value interpolated at Z = 0 
        for i in reg.coords["x"].values:
            ring = reg.sel(x=i)
            Z = ring["Z"].values
            
            # Put parameters into dataframe
            for param in ["dr"] + params:
                
                if param in ring:
                    interp = scipy.interpolate.interp1d(Z, ring[param].values, kind = "quadratic")
                    df.loc[i, param] = interp(0)
                else:
                    print(f"Parameter {param} not found")
            
        df.reset_index(inplace = True, drop = True)
    
    else:
        # Take region directly from named selection
        if region in ["inner_lower_target", "inner_upper_target", "outer_lower_target", "outer_upper_target"]:
            reg = ds.hermesm.select_region(region).squeeze()
        
        # Get data by poloidal index
        elif poloidal_index is None:
            raise Exception("If not requesting midplane, please pass poloidal_index")
        else:
            reg = ds.isel(x = xslice, theta = poloidal_index).squeeze()
    
        df["dr"] = reg["dr"].values

        for param in params:
            if param in reg:
                df[param] = reg[param].values
            else:
                print(f"Parameter {param} not found")
    
    ## Calculate radial distance from separatrix
    for i, _ in enumerate(df["dr"]):
        if i == 0:
            df.loc[i, "Srad"] = df.loc[i, "dr"] / 2
        else:
            df.loc[i, "Srad"] = df.loc[i-1, "Srad"] + df.loc[i-1, "dr"]/2 + df.loc[i, "dr"]/2
        
    df["sep"] = 0
    
    sepind = ds.metadata["ixseps1g"]
    df.loc[sepind, "sep"] = 1
    dfsep = df[df["sep"] == 1]
    
    # Correct so 0 is inbetween the cell centres straddling separatrix
    sepcorr = (df["Srad"][sepind] - df["Srad"][sepind-1]) / 2

    df["Srad"] -= dfsep["Srad"].values - sepcorr
    
    ## Cut off SOL or core if necessary
    if sol is False and core is False:
        raise Exception("Please specify sol or core to be True")
    
    if sol is False:
        df = df[df["Srad"] < 0]
    
    if core is False:
        df = df[df["Srad"] > 0]    
        
    return df

def _interpolate_exact_sol_ring(
    ds,
    params,
    region,
    sepdist,
    debug = False,
    ):
    """
    This returns poloidal data radially interpolated to the same psi value as a desired
    separatrix distance at the midplane. This means the data is not affected by radial resolution.
    Good for code comparisons.
    
    Parameters
    ----------
    ds : Dataset to use (must be loaded with guards)
    params : list of parameters to extract
    region : string, region to extract data from (inner_lower, inner_upper, outer_lower, outer_upper)
    sepdist : float, distance from separatrix to extract data from
    
    Returns
    -------
    df : DataFrame with data along the field line to be used by get_1d_poloidal_data
    """

    
    m = ds.metadata
    extra_params = ["R", "Z", "psi_poloidal", "dpol", "Bxy", "Bpxy"]  # Needed in get_1d_poloidal_data
    all_params = extra_params + params

    ## This comes from _select_custom_sol_ring
    start, end = _get_poloidal_range(ds, region)   

    ## Need to choose the psi we want based on an accurate midplane slice
    if "outer" in region:
        start_region = "omp"
    elif "inner" in region:
        start_region = "imp"
    else:
        raise Exception("Region not recognised, needs to mention inner or outer")

    midplane = get_1d_radial_data(ds, params = all_params, region = start_region, core = False)

    # Find exact psi at chosen separatrix distance
    psi = scipy.interpolate.interp1d(midplane["Srad"], midplane["psi_poloidal"])(sepdist)

    ## Interpolate data on the same psi and return
    df = pd.DataFrame()
    if debug:
        fig, ax = plt.subplots()
    for i in range(start,end):
        radial = get_1d_radial_data(ds, params = all_params, poloidal_index = i, core = False)
        
        if debug:
            ds["Td"].bout.polygon(ax = ax, grid_only = True, linecolor = "k", linewidth = 0.1, antialias = True, separatrix = False, add_colorbar = False)
            ax.plot(radial["R"], radial["Z"], "o", markersize = 3, alpha = 0.5)
        
        for param in all_params:
            df.loc[i, param] = scipy.interpolate.interp1d(radial["psi_poloidal"], radial[param])(psi)
            
    df = df.reset_index(drop=True)
    
    return df


def get_1d_poloidal_data(
    ds, 
    params, 
    region, 
    sepdist = None, 
    sepadd = None,
    target_first = False,
    interpolate_radial = False,
    interpolate_poloidal = False,
    interpolate_poloidal_resolution = 100):
    """
    Return a dataframe with data along a field line as well as poloidal and parallel connection lengths.
    Refer to get_custom_sol_ring for the indexing routine. 
    Select the field line by either sepdist or sepadd, but not both.
    Only supports one divertor leg at a time. 
    
    
    Parameters
    ----------
    ds : Dataset to use (must be loaded with guards)
    params : list of parameters to extract
    region : string, region to extract data from (inner_lower, inner_upper, outer_lower, outer_upper)
    sepdist : float, distance from separatrix to extract data from
    sepadd : int, SOL ring index to extract data from (starting from one after separatrix)
    target_first : bool, if True, reverse the dataframe so that 0 distance is at the target

    """
    if "t" in ds.sizes:
        raise Exception("get_1d_poloidal_data doesn't support multiple time slices")
    
    
    ## Select SOL region
    # This is always in index ascending order, and goes one point past the midplane
    
    # 1. Get it at an arbitrary sepdist using radial interpolation 
    if interpolate_radial:
        if sepdist is None:
            raise Exception("sepdist must be specified if interpolate_radial is True")
        
        df = _interpolate_exact_sol_ring(
            ds, 
            params, 
            region, 
            sepdist = sepdist, 
            debug = False
        )
    
    # 2. Or get it at a specific SOL ring index, or the closest SOL ring to a given sepdist
    else:
        reg = ds.hermesm.select_custom_sol_ring(region, sepadd = sepadd, sepdist = sepdist).squeeze()

        df = pd.DataFrame()
        df["R"] = reg["R"].values
        df["Z"] = reg["Z"].values
        df["dpol"] = reg["dpol"].values  # Poloidal cell width
        df["Bxy"] = reg["Bxy"].values  # Total magnetic field
        df["Bpxy"] = reg["Bpxy"].values  # Poloidal magnetic field
        
        for param in params:
            if param in reg:
                df[param] = reg[param].values
            else:
                print(f"Parameter {param} not found")
    
    # Calculate geometry bits
    df["dpar"] = df["dpol"] * abs(df["Bxy"] / abs(df["Bpxy"]))  # Parallel cell width
    df["Spol"] = df["dpol"].cumsum()   # Poloidal connection length
    df["Spar"] = df["dpar"].cumsum()   # Parallel connection length


    # Optionally increase poloidal/parallel resolution            
    if interpolate_poloidal:
        df_interp = pd.DataFrame()
        Z = df["Z"].values
        R = df["R"].values
        ds = np.sqrt(np.diff(R)**2 + np.diff(Z)**2)
        s = np.concatenate(([0], np.cumsum(ds)))  # shape (N,)
        
        # R_spline = scipy.interpolate.UnivariateSpline(s, R, s=0)
        # Z_spline = scipy.interpolate.UnivariateSpline(s, Z, s=0)
        
        # CubicSpline better than UniveriateSpline cause it allows boundaries.
        # Setting it to clamped helps match at edges.
        R_spline = scipy.interpolate.CubicSpline(s, R, bc_type = "clamped", extrapolate = False)
        Z_spline = scipy.interpolate.CubicSpline(s, Z, bc_type = "clamped", extrapolate = False)
        
        s_fine = np.linspace(s[0], s[-1], interpolate_poloidal_resolution)
        df_interp["R"] = R_spline(s_fine)
        df_interp["Z"] = Z_spline(s_fine)
        
        for param in df.columns.drop(["R", "Z"]):
            # param_spline = scipy.interpolate.UnivariateSpline(s, df[param].values, s=0)
            param_spline = scipy.interpolate.CubicSpline(s, df[param].values, bc_type = "clamped", extrapolate = False)
            df_interp[param] = param_spline(s_fine)
            
        df = df_interp
        
        
    # Flip outer upper and inner lower so that they start at midplane
    if any([x in region for x in ["outer_upper", "inner_lower"]]):
        df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
        df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
        df = df.iloc[::-1].reset_index(drop = True)
        
    # Check and delete any extraneous points upstream of the midplane
    signs = np.sign(df["Z"])
    sign_changes = signs != signs.shift()
    change_indices = df.index[sign_changes][1:]
    if len(change_indices) > 1:
        raise Exception("Multiple sign changes in Z. Haven't considered this yet")

    idx_before_mp = change_indices[0]-1
    df = df.iloc[idx_before_mp:].reset_index(drop = True)

    # Interpolate start of field line onto Z = 0  (between midplane_a and midplane_b)
    # Assumes only one point upstream of midplane
    for param in df.columns.drop("Z"):
        interp = scipy.interpolate.interp1d(df["Z"], df[param], kind = "linear")
        df.loc[0, param] = interp(0)
    df.loc[0,"Z"] = 0 
    df["Spol"] -= df["Spol"].iloc[0]   # Now 0 is at Z = 0
    df["Spar"] -= df["Spar"].iloc[0]   # Now 0 is at Z = 0
            
    # Flip data so target is first if needed
    if target_first:
        df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
        df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
        df = df.iloc[::-1]
        
    return df