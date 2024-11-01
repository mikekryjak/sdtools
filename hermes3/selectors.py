from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
from hermes3.plotting import *
from hermes3.front_tracking import *
import numpy as np
import scipy

def get_1d_radial_data(ds, params, region, average_midplanes = False):
    """
    Return a Pandas Dataframe with a radial slice of data in the given region
    The dataframe contains a radial distance normalised to the separatrix
    Guards are excluded. if OMP or IMP is selected, it will interpolate onto Z = 0
    """

    if "midplane" in region:
        if not any([x in region for x in ["_a", "_b"]]):
            region = region + "_a"
        
    df = pd.DataFrame()
    
    # Interpolate to Z = 0 for IMP or OMP
    if region == "omp" or region == "imp":
        m = ds.metadata
        if region == "omp":
            reg = ds.isel(x = slice(m["MXG"], -m["MXG"]), theta = slice(m["omp_a"] - 2, m["omp_b"] + 2))
        else:
            reg = ds.isel(x = slice(m["MXG"], -m["MXG"]),theta = slice(m["imp_a"] - 2, m["imp_b"] + 2))
        
        
        for i in reg.coords["x"].values:
            ring = reg.sel(x=i)
            Z = ring["Z"].values
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
            df.loc[i, "Srad"] = df.loc[i-1, "Srad"] + df.loc[i-1, "dr"] + df.loc[i, "dr"]/2
        
    df["sep"] = 0
    
    sepind = ds.metadata["ixseps1g"]
    df.loc[sepind, "sep"] = 1
    dfsep = df[df["sep"] == 1]
    
    # Correct so 0 is inbetween the cell centres straddling separatrix
    sepcorr = (df["Srad"][sepind] - df["Srad"][sepind-1]) / 2

    df["Srad"] -= dfsep["Srad"].values - sepcorr

    
        
    return df


def get_1d_poloidal_data(ds, params, region, sepdist = None, sepadd = None, target_first = False):
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
    print(sepadd, sepdist)
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
        
    # Ensure inner starts at midplane
    if "inner" in region:
        df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
        df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
        df = df.iloc[::-1].reset_index(drop = True)

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