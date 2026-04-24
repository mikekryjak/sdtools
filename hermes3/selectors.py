from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
from hermes3.plotting import *
from hermes3.front_tracking import *
import xhermes
import numpy as np
import pandas as pd
import scipy
import xarray as xr

## Deprecated by xhermes
# poloidal slices now found in ds.metadata["poloidal_slices"]
# def _get_poloidal_range(ds, region):
#     """
#     Returns poloidal region start and end indices in a tuple.
#     Each region is a quarter of a CDN topology corresponding to a 
#     divertor. The regions are all in a positive index order.
#     The regions extend past the midplane by one point.
#     """
#     m = ds.metadata
    
#     if m["topology"] == "connected-double-null":

#         ## This comes from _select_custom_sol_ring
#         if region == "inner_lower_sol":
#             start = m["MYG"]
#             end = m["imp_a"] + 1
#         elif region == "inner_upper_sol":
#             start = m["imp_b"]
#             end = m["ny_inner"] + m["MYG"]
#         elif region == "outer_lower_sol":
#             start = m["omp_a"]
#             end = m["nyg"] - m["MYG"]
#         elif region == "outer_upper_sol":
#             start = m["ny_inner"] + m["MYG"]*3
#             end = m["omp_b"]+1
#         elif region == "inner_sol":
#             start = m["MYG"]
#             end = m["ny_inner"] + m["MYG"]
#         elif region == "outer_sol":
#             start = m["ny_inner"] + m["MYG"] * 3
#             end = m["nyg"] - m["MYG"]
#         elif region == "inner_core":
#             start = m["j1_1g"]+1
#             end = m["j2_1g"]+1
#         elif region == "outer_core":
#             start = m["j1_2g"]+1
#             end = m["j2_2g"]+1

#         else:
#             raise ValueError(f"Unknown region {region} for poloidal range")
        
#     else:
#         raise ValueError(f"Topology {m['topology']} not yet supported")
    
#     return (start,end)

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

    # ---- helpers ----
    def _quad_at_zero(Z1d, V1d):
        """
        Quadratic interpolate V(Z) to Z=0 using the three closest Z points.
        Robust to non-monotonic Z; falls back to linear/nearest if needed.
        """
        Z1d = np.asarray(Z1d)
        V1d = np.asarray(V1d)
        if Z1d.size == 0:
            return np.nan
        n = min(3, Z1d.size)
        k = np.argpartition(np.abs(Z1d), n - 1)[:n]
        kz = k[np.argsort(Z1d[k])]
        z = Z1d[kz]
        y = V1d[kz]

        if z.size == 1:
            return float(y[0])
        if z.size == 2 or np.any(np.isclose(np.diff(z), 0.0)):
            return float(np.interp(0.0, z, y))

        denom0 = (z[0]-z[1])*(z[0]-z[2])
        denom1 = (z[1]-z[0])*(z[1]-z[2])
        denom2 = (z[2]-z[0])*(z[2]-z[1])
        if np.isclose(denom0,0) or np.isclose(denom1,0) or np.isclose(denom2,0):
            return float(np.interp(0.0, z, y))
        w0 = (z[1]*z[2]) / denom0
        w1 = (z[0]*z[2]) / denom1
        w2 = (z[0]*z[1]) / denom2
        return float(y[0]*w0 + y[1]*w1 + y[2]*w2)

    # ---- start of original logic ----
    df = pd.DataFrame()
    m = ds.metadata
    
    xslice = slice(None, None)
        
    if region is None and poloidal_index is None:
        raise Exception("Please specify region or poloidal_index")
    
    translate_dict = {
        "outer_midplane" : "omp",
        "inner_midplane" : "imp"
    }
    if region in translate_dict:
        region = translate_dict[region]
    
    # Interpolate to Z = 0 for IMP or OMP
    if region == "omp" or region == "imp":

        omp_a = xhermes.selector_poloidal(ds, "outer_upper_midplane")
        omp_b = xhermes.selector_poloidal(ds, "outer_lower_midplane")
        imp_a = xhermes.selector_poloidal(ds, "inner_upper_midplane")
        imp_b = xhermes.selector_poloidal(ds, "inner_lower_midplane")
        
        # slice a narrow band around the midplane
        if region == "omp":
            reg = ds.isel(x=xslice, theta=slice(omp_a - 2, omp_b + 2))
        else:
            reg = ds.isel(x=xslice, theta=slice(imp_a - 2, imp_b + 2))

        # Ensure a single chunk along the core dim for gufunc
        # (cheap here since theta band is small)
        reg = reg.unify_chunks()
        try:
            reg = reg.chunk(dict(theta=-1))
        except Exception:
            # ok if already NumPy-backed or chunking not applicable
            pass

        # warn on missing vars (keep behavior)
        for param in params:
            if param not in ds:
                print(f"Parameter {param} not found")

        # build list of variables to interpolate (include dr)
        wanted = ["dr"] + [p for p in params if p in ds]
        # vectorized quadratic-at-zero for each var across all x
        out = {}
        for p in wanted:
            out[p] = xr.apply_ufunc(
                _quad_at_zero,
                reg["Z"], reg[p],
                input_core_dims=[["theta"], ["theta"]],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                dask_gufunc_kwargs={"allow_rechunk": True},  # Each core dim must be single chunk
                output_dtypes=[float],
            )
        mid = xr.Dataset(out)

        # to pandas, matching previous behavior (drop x index -> 0..N-1)
        df = mid.to_dataframe().reset_index(drop=True)
        df = df.drop(columns=["t"], errors="ignore")

    else:

        if poloidal_index is not None and region is not None:
            raise Exception("Please specify only one of region or poloidal_index, not both")

        if poloidal_index is not None:
            reg = ds.isel(x=xslice, theta=poloidal_index).squeeze()

        # Take region directly from named selection
        if region is not None:
            if region in xhermes.selector_poloidal(ds, return_available = True):
                reg = ds.hermes.select_region(radial_region = "domain_guards", poloidal_region = region).squeeze()
            else:
                raise ValueError(f"Unknown region {region}.")
            
    
        df["dr"] = reg["dr"].values

        for param in params:
            if param in reg:
                df[param] = reg[param].values
            else:
                print(f"Parameter {param} not found")
    

    # Calculate radial distance from separatrix (vectorized)
    dr = df["dr"].to_numpy()
    df["Srad"] = np.cumsum(dr) - 0.5 * dr
    
    df["sep"] = 0
    sepind = ds.metadata["ixseps1"]
    df.loc[sepind, "sep"] = 1
    dfsep = df[df["sep"] == 1]
    
    # Correct so 0 is in between the cell centres straddling separatrix
    sepcorr = (df["Srad"][sepind] - df["Srad"][sepind - 1]) / 2
    df["Srad"] -= dfsep["Srad"].values - sepcorr

    # Label regions
    df["region"] = ""
    df.loc[df["Srad"] > 0, "region"] = "sol"
    df.loc[df["Srad"] < 0, "region"] = "core"

    df["radial_index"] = df.index.values

    # Remove guards if necessary
    # NOTE: you must preserve the original index here!
    if not guards:
        df = df.iloc[m["MXG"] : -m["MXG"]]
    
    # Cut off SOL or core if necessary
    if sol is False and core is False:
        raise Exception("Please specify sol or core to be True")
    
    if sol is False:
        df = df[df["Srad"] < 0]
    
    if core is False:
        df = df[df["Srad"] > 0]    
    
    return df

def get_1d_radial_data_old(ds, 
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
       
       # Take 2 cells on either side to then interpolate in the middle
        if region == "omp":
            reg = ds.isel(x = xslice, theta = slice(m["omp_a"] - 2, m["omp_b"] + 2))
        else:
            reg = ds.isel(x = xslice, theta = slice(m["imp_a"] - 2, m["imp_b"] + 2))
            
        for param in params:
            if param not in ds.data_vars:
                print(f"Parameter {param} not found")
        
        # For every parameter, collect the value interpolated at Z = 0 
        for i in reg.coords["x"].values:
            ring = reg.sel(x=i)
            Z = ring["Z"].values
            
            # Put parameters into dataframe
            for param in ["dr"] + params:
                if param in ring:
                    interp = scipy.interpolate.interp1d(Z, ring[param].values, kind = "quadratic")
                    df.loc[i, param] = interp(0)
                
            
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

def _interpolate_exact_poloidal_ring(
    ds,
    params,
    region,
    sepadd = None,
    sepdist = None,
    radial_start_region="auto",
    debug = False,
    ):
    """
    This returns poloidal data radially interpolated to the same psi value as a desired
    separatrix distance at the midplane. This means the data is not affected by radial resolution.
    Good for code comparisons.

    NOTE: Only poloidally contiguous regions are supported, i.e. those that you can get with
    _get_poloidal_range.
    
    Parameters
    ----------
    ds : Dataset to use (must be loaded with guards)
    params : list of parameters to extract
    region : string, region to extract data from (inner_lower, inner_upper, outer_lower, outer_upper)
    radial_start_region : string, region to use for the radial slice on which separatrix distance is defined
    sepdist : float, distance from separatrix to extract data from (use either this or sepadd, not both)
    sepadd : int, SOL ring index to extract data from (starting from one after separatrix)
    
    Returns
    -------
    df : DataFrame with data along the field line to be used by get_1d_poloidal_data
    """

    if region == "all":
        region = "sol"
    elif region in ["inner", "outer", "inner_lower", "inner_upper", "outer_lower", "outer_upper"]:
        region = f"{region}_sol"

    if any([name in region for name in ["sol", "upstream", "divertor"]]):
        if sepdist is not None and sepdist < 0:
            raise ValueError("sepdist must be positive for SOL regions")
        if sepadd is not None and sepadd < 0:
            raise ValueError("sepadd must be positive for SOL regions")
    else:
        if sepdist is not None and sepdist > 0:
            raise ValueError("sepdist must be negative for core or PFR regions")
        if sepadd is not None and sepadd > 0:
            raise ValueError("sepadd must be negative for core or PFR regions")

    
    m = ds.metadata
    extra_params = ["R", "Z", "psi_poloidal", "dpol", "Bxy", "Bpxy"]  # Needed in get_1d_poloidal_data
    all_params = extra_params + params

    ## This comes from _select_custom_sol_ring
    # start, end = _get_poloidal_range(ds, region)   
    poloidal_selection = xhermes.selector_poloidal(ds, region)
    poloidal_indices = ds["theta_idx"].values[poloidal_selection]

    ## Get radial slice to figure out field line starting point based on sepdist
    if radial_start_region == "auto":
        if region == "outer_lower_pfr":
            radial_start_region = "outer_lower_target"
        elif region == "outer_upper_pfr":
            radial_start_region = "outer_upper_target"
        elif region == "inner_lower_pfr":
            radial_start_region = "inner_lower_target"
        elif region == "inner_upper_pfr":
            radial_start_region = "inner_upper_target"
        elif "outer" in region:
            radial_start_region = "omp"
        elif "inner" in region:
            radial_start_region = "imp"
        elif region == "lower_pfr":
            radial_start_region = "outer_lower_target"
        elif region == "upper_pfr":
            radial_start_region = "outer_upper_target"

    radial_slice = get_1d_radial_data(ds, params = all_params, region = radial_start_region, core = True)

    # Find exact psi at chosen separatrix distance
    if sepdist is not None:
        psi = scipy.interpolate.interp1d(radial_slice["Srad"], radial_slice["psi_poloidal"])(sepdist)
    elif sepadd is not None:
        psi = radial_slice.loc[radial_slice.index[sepadd], "psi_poloidal"]
    else:
        raise ValueError("Either sepdist or sepadd must be provided")

    ## Interpolate data on the same psi and return
    df = pd.DataFrame()
    if debug:
        fig, ax = plt.subplots(dpi = 150)
        ds["Td"].bout.polygon(ax = ax, grid_only = True, linecolor = "k", linewidth = 0.1, antialias = True, separatrix = False, add_colorbar = False)

    # For every poloidal index, get the radial data, interpolate to the correct sepdist and append to dataframe
    for i in poloidal_indices:
        radial = get_1d_radial_data(ds, params = all_params, poloidal_index = i, core = True)
        
        if debug:
            ax.plot(radial["R"], radial["Z"], "o", markersize = 3, alpha = 0.5)
        
        for param in all_params:
            df.loc[i, param] = scipy.interpolate.interp1d(radial["psi_poloidal"], radial[param])(psi)
            
    df = df.reset_index(drop=True)

    if debug:
        ax.plot(df["R"], df["Z"], c = "deeppink")
        ax.set_title("")
        
    
    return df

def get_1d_poloidal_data_double(
    ds,
    params,
    region="outer",
    **kwargs,
):
    """
    Prepare a full two-divertor field line dataset by using get_1d_poloidal_data
    for both upper and lower and concatenating them.

    Each dataset starts at the midplane with an interpolated value exactly at Z=0,
    so here we flip the upper leg and join it to the lower leg while removing one
    of the duplicate midplane points.
    """

    if "lower" in region or "upper" in region:
        raise ValueError(
            "Region should be 'inner' or 'outer' for double leg data, not 'inner_lower' etc."
        )

    df_lower = get_1d_poloidal_data(ds, params, region=f"{region}_lower", **kwargs)
    df_upper = get_1d_poloidal_data(ds, params, region=f"{region}_upper", **kwargs)
    fl = pd.concat([df_upper.iloc[::-1], df_lower.iloc[1:]], ignore_index=True)

    return fl

def get_1d_poloidal_data(
    ds, 
    params, 
    region, 
    sepdist = None, 
    sepadd = None,
    guards = False, 
    target_first = False,
    interpolate_midplane = True,
    interpolate_radial = False,
    radial_start_region = None,
    interpolate_poloidal = False,
    interpolate_poloidal_resolution = 100,
    debug = False):
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
    if region == "all":
        region = "sol"
    elif region in ["inner", "outer", "inner_lower", "inner_upper", "outer_lower", "outer_upper"]:
        region = f"{region}_sol"

    m = ds.metadata

    if "t" in ds.sizes:
        raise Exception("get_1d_poloidal_data doesn't support multiple time slices")
    
    if "core" in region and any([target_first, interpolate_midplane]):
        raise Exception("target_first and interpolate_midplane are incompatible with region=core")
    
    if "divertor" in region and interpolate_midplane:
        raise ValueError("Interpolate midplane should not be used for divertor regions!")

    if any([x in region for x in ["pfr", "core"]]) and interpolate_midplane:
        raise ValueError("Interpolate midplane should not be used for PFR or core regions!")
    

    ## Select SOL region
    # This is always in index ascending order, and goes one point past the midplane
    
    # 1. Get it at an arbitrary sepdist using radial interpolation 
    if interpolate_radial:
        if sepdist is None:
            raise Exception("sepdist must be specified if interpolate_radial is True")
        
        df = _interpolate_exact_poloidal_ring(
            ds, 
            params, 
            region, 
            sepdist = sepdist, 
            radial_start_region = radial_start_region,
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
        df["theta_idx"] = reg["theta_idx"].values.astype(int)  # Poloidal index

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
        
        
    # Flip outer upper and inner lower so that the regions have a
    # consistent orientation across all legs.
    if any([name in region for name in [
        "outer_upper_sol", "inner_lower_sol",
        "outer_upper_upstream", "inner_lower_upstream",
        "outer_upper_divertor", "inner_lower_divertor",
        "outer_upper_pfr", "inner_lower_pfr"
        ]]):
        df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
        df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
        df = df.iloc[::-1].reset_index(drop = True)
        
    # Check and delete any extraneous points upstream of the midplane
    if interpolate_midplane:
        signs = np.sign(df["Z"])
        sign_changes = signs != signs.shift()
        change_indices = df.index[sign_changes][1:]
        if len(change_indices) > 1:
            raise Exception("Multiple sign changes in Z. Haven't considered this yet")

        idx_before_mp = change_indices[0]-1
        df = df.iloc[idx_before_mp:].reset_index(drop = True)

    # Interpolate start of field line onto Z = 0  (between midplane_a and midplane_b)
    # Assumes only one point upstream of midplane
    if interpolate_midplane:
        for param in df.columns.drop(["Z", "theta_idx"]):
            interp = scipy.interpolate.interp1d(df["Z"], df[param], kind = "linear")
            df.loc[0, param] = interp(0)
        df.loc[0,"Z"] = 0 
        df.loc[0, "theta_idx"] = np.nan
        df["Spol"] -= df["Spol"].iloc[0]  # Now 0 is at Z = 0
        df["Spar"] -= df["Spar"].iloc[0]  # Now 0 is at Z = 0


    if "sol" in region:
        xpoint_index_name = f"{region.replace("_sol", "")}_xpoint"
        global_xpoint_index = xhermes.selector_poloidal(ds, xpoint_index_name)

        Xpoint_index = df[df["theta_idx"] == global_xpoint_index].index[0]

        # df["Xpoint"] = 0
        # df.loc[Xpoint_index, "Xpoint"] = 1

        df.loc[:Xpoint_index, "region"] = "upstream"
        df.loc[Xpoint_index:, "region"] = "divertor"
    elif "upstream" in region:
        df["region"] = "upstream"
    elif "divertor" in region:
        df["region"] = "divertor"
    elif "pfr" in region:
        df["region"] = "pfr"
    elif "core" in region:
        df["region"] = "core"
    else:
        raise ValueError(f"Unknown region {region} for region labeling")
    
    # Take out guard cell if needed
    if not guards and any([x in region for x in ["sol", "divertor", "pfr"]]):
        df = df.iloc[:-1]
            
    # Flip data so target is first if needed
    if target_first:
        df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
        df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
        df = df.iloc[::-1]
        
    return df