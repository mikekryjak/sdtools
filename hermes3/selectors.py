from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
from hermes3.plotting import *
from hermes3.front_tracking import *
import numpy as np
import scipy

def get_1d_radial_data(ds, params, region, guards = False):
    """
    Return a Pandas Dataframe with a radial slice of data in the given region
    The dataframe contains a radial distance normalised to the separatrix
    Guards are excluded. if OMP or IMP is selected, it will interpolate onto Z = 0
    """

    if "midplane" in region:
        if not any([x in region for x in ["_a", "_b"]]):
            region = region + "_a"
        
    df = pd.DataFrame()
    m = ds.metadata
    
    if guards:
        xslice = slice(None, None)
    else:
        xslice = slice(m["MXG"], -m["MXG"])
    
    # Interpolate to Z = 0 for IMP or OMP
    if region == "omp" or region == "imp":
       
        
        
        if region == "omp":
            reg = ds.isel(x = xslice, theta = slice(m["omp_a"] - 2, m["omp_b"] + 2))
        else:
            reg = ds.isel(x = xslice,theta = slice(m["imp_a"] - 2, m["imp_b"] + 2))
        
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
    
    # Take region directly
    else:
        reg = ds.hermesm.select_region(region).squeeze()
        df["dr"] = reg["dr"].values
        
        for param in params:
            if param in reg:
                df[param] = reg[param].values
            else:
                print(f"Parameter {param} not found")

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

    
        
    return df


def get_1d_poloidal_data(
    ds, 
    params, 
    region, 
    sepdist = None, 
    sepadd = None,
    target_first = False,
    interpolate = False):
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
    if "t" in ds.dims:
        raise Exception("get_1d_poloidal_data doesn't support multiple time slices")
    reg = ds.hermesm.select_custom_sol_ring(region, sepadd = sepadd, sepdist = sepdist).squeeze()

    df = pd.DataFrame()
    df["Z"] = reg["Z"].values
    df["dpar"] = reg["dpol"] * abs(reg["Bxy"] / abs(reg["Bpxy"]))  # Parallel cell width
    df["Spol"] = reg["dpol"].cumsum()   # Poloidal connection length
    df["Spar"] = df["dpar"].cumsum()   # Parallel connection length

    for param in params:
        if param in reg:
            df[param] = reg[param].values
        else:
            print(f"Parameter {param} not found")
            
    if interpolate:
        df_interp = pd.DataFrame()
        Z = df["Z"].values
        R = df["R"].values
        ds = np.sqrt(np.diff(R)**2 + np.diff(Z)**2)
        s = np.concatenate(([0], np.cumsum(ds)))  # shape (N,)
        R_spline = scipy.interpolate.UnivariateSpline(s, R, s=0)
        Z_spline = scipy.interpolate.UnivariateSpline(s, Z, s=0)
        
        
        s_fine = np.linspace(s[0], s[-1], 100)
        df_interp["R"] = R_spline(s_fine)
        df_interp["Z"] = Z_spline(s_fine)
        
        for param in df.columns.drop(["R", "Z"]):
            param_spline = scipy.interpolate.UnivariateSpline(s, df[param].values, s=0)
            df_interp[param] = param_spline(s_fine)
            
        df = df_interp
        
        
    # Ensure inner starts at midplane
    if "inner" in region:
        df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
        df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
        df = df.iloc[::-1].reset_index(drop = True)
        
    # Keep only one point beyond midplane
    signs = np.sign(df["Z"])
    sign_changes = signs != signs.shift()
    change_indices = df.index[sign_changes][1:]
    if len(change_indices) > 1:
        raise Exception("Multiple sign changes in Z. Haven't considered this yet")

    idx_before_mp = change_indices[0]-1
    df = df.iloc[idx_before_mp:].reset_index(drop = True)

    # Interpolate start of field line onto Z = 0  (between midplane_a and midplane_b)
    for param in df.columns.drop("Z"):
        interp = scipy.interpolate.interp1d(df["Z"], df[param], kind = "linear")
        df.loc[0, param] = interp(0)
    df.loc[0,"Z"] = 0  

    df["Spol"] -= df["Spol"].iloc[0]   # Now 0 is at Z = 0
    df["Spar"] -= df["Spar"].iloc[0]   # Now 0 is at Z = 0
            
    if target_first:
        df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
        df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
        df = df.iloc[::-1]
        
    return df