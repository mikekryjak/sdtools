
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from fusiondls.postprocessing import FrontLocation, FrontLocationScan
# from ThermalFrontFormulation import *

class DLSoutput():
    """
    Result of a single DLS front position
    """
    
    def __init__(self, profile, result):
        self.p = profile
        self.r = result
        

def plot_profile_histories(p, s, list_idx, title = "", 
                           rad_threshold = 0.7, 
                           mode = "temp", 
                           axis = "parallel",
                           use_prad = True   # Use radiation in W rather than W/m2 for front width?
                           ):  
          
    fig, axes = plt.subplots(1,4,figsize = (15*len(list_idx)/1,4))
    
    axstore = {}

    for i, idx in enumerate(list_idx):
        axstore[i] = {} 
        ax1 = axstore[i]["ax1"] = axes[i]
        
        if axis == "parallel":
            x = s["Sprofiles"][idx]
            ax1.set_xlabel(r"$L_{\parallel}$[m]")
        elif axis == "poloidal":
            x = s["SPolprofiles"][idx]
            ax1.set_xlabel(r"$L_{\theta}$[m]")
        else:
            raise ValueError("axis must be 'parallel' or 'poloidal'")
        
        Xpoint = s["Xpoints"][idx]
        
        R = s["Rprofiles"][idx]
        R = pad_profile(x, R)
        
        q = pad_profile(x, s["Qprofiles"][idx])  # W/m2
        Btot = np.array(s["Btotprofiles"][idx])  # T
        Bpol = np.array(s["Bpolprofiles"][idx])  # T
        T = pad_profile(x, s["Tprofiles"][idx])
        P = T[-1] * s["cvar"][idx]
        N = P/T

        ## Radiated power in W/m2 (includes B effects)
        Rcum = sp.integrate.cumulative_trapezoid(y = R, x = x, initial = 0)
        Rcum /= Rcum.max()
        
        # qradial is the uniform upstream heat source
        qradial = np.ones_like(x)
        qradial[Xpoint:] = s["state"].qradial
        
        ## Radiated power in W (excludes B effects)
        Prad = np.gradient(q/Btot, x) + qradial/Btot
        Pradcum = sp.integrate.cumulative_trapezoid(y = Prad, x = x, initial = 0)
        Pradcum /= Pradcum.max()
        
        if use_prad is True:
            Rchoice = Pradcum
        else:
            Rchoice = Rcum
        
        front_start_idx = np.argmin(abs(Rchoice - Rchoice[Rchoice> 0.01*Rchoice.max()][0]))
        front_end_idx = np.argmin(abs(Rchoice - rad_threshold))

        ## Axis 1, radiation
        # ax.plot(x, R, label = "Radiation", marker = "o", ms = 0, lw = 2)
        # ax.set_ylabel("Radiation [Wm-3]")
        # ax.spines["left"].set_color("teal")
        # ax.tick_params(axis = "y", colors = "teal")
        # ax.yaxis.label.set_color(color = "teal")
        ax1.set_yscale("log")
        ax1.spines["top"].set_visible(True)
        
        # ax1.axvline(x = x[Xpoint], c = "k", ls = ":", lw = 1.5)

        ## Axis 2, radiation integral
        ax2 = axstore[i]["ax2"] = ax1.twinx()
        c = "blue"
        ax2.plot(x, Rchoice, color = c, ls = "-", label = "Cumulative radiation integral", lw = 2,   alpha = 1)
        ax2.set_ylabel("Cumulative radiation fraction")
        ax2.spines["right"].set_color(c)
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(axis = "y", colors = c)
        ax2.yaxis.label.set_color(color = c)
        ax2.fill_between(x = [x[front_start_idx], x[front_end_idx]], y1 =0, y2 = 1, color = c, hatch = "//", alpha = 0.1)
        ax2.set_ylim(0,1.1)

        ## Axis 3, temperature
        ax3 = axstore[i]["ax3"] = ax1.twinx()
        if i > 0:
            ax1.sharey(axstore[i-1]["ax1"])
            ax2.sharey(axstore[i-1]["ax2"])
            ax3.sharey(axstore[i-1]["ax3"])
        
        if mode == "temp":
            ax3color = "red"
            ax3.plot(x, pad_profile(x, s["Tprofiles"][idx]), c = ax3color, lw = 2, label = "Temperature", alpha = 1)
            ax3.set_ylabel("T")
            ax3.set_ylim(0, None)
        
        elif mode == "qpar":
            ax3color = "deeppink"
            ax3.plot(x, pad_profile(x, s["Qprofiles"][idx]), c = ax3color, lw = 2, label = r"$q_{\parallel}$", alpha = 1)
            ax3.set_ylabel("qpar")
            ax3.set_ylim(0,None)
            
        elif mode == "Btot":
            ax3color = "teal"
            ax3.plot(x, pad_profile(x, p["Btot"]), c = ax3color, lw = 2, label = r"$B_{tot}$", alpha = 1)
            ax3.set_ylabel("Btot [T]")
            ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))
            
        elif mode == "rad":
            ax3color = "darkorange"
            ax3.plot(x, pad_profile(x, s["Rprofiles"][idx]), c = ax3color, lw = 2, label = r"$R$", alpha = 1)
            ax3.set_ylabel("R [Wm-3]")
            ax3.set_ylim(0, None)
            
            
            # ax3.set_ticklabel_format(useOffset = False, axis = "y", style = "plain")
            # ax3.set_ylim(0,None)
            
        else:
            raise ValueError("mode must be 'temp' or 'qpar'")

        ax3.spines["right"].set_position(("outward", 75))
        ax3.spines["right"].set_visible("True")
        ax3.spines["right"].set_color(ax3color)
        ax3.tick_params(axis = "y", colors = ax3color, direction = "in")
        ax3.yaxis.label.set_color(color = ax3color)
        

        
        # fig.legend(loc = "lower center", bbox_to_anchor = (0.6,0.9), ncol = 2)

        axstore[i]["ax1"] = ax1
        axstore[i]["ax2"] = ax2
        axstore[i]["ax3"] = ax3
        
        # Disable RHS Y labels in all but the last plot
        for ax in [axstore[i]["ax2"], axstore[i]["ax3"]]:
            if i < len(list_idx)-1:
                # ax.set_yticklabels([])
                for label in ax.get_yticklabels():
                    label.set_visible(False)

                ax.set_ylabel("")
                ax.spines["right"].set_visible(False)
                
                ax.tick_params(axis="y", which = "both", right = False)
                 
        if mode == "qpar":   
            axstore[i]["ax3"].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1e}"))
                    # ax.set_ticklabel_format(useOffset = False, axis = "y", style = "plain")
                    # for tick in ax.get_yticks():
                    #     tick.set_visible(False)
                    # ax.set_yticks([])
                    
        # Disable LHS Y labels in all but the first plot
        if i > 0 :
            axstore[i]["ax1"].set_yticklabels([])
            
        axstore[i]["ax1"].grid(which = "both", visible = False)
        axstore[i]["ax2"].grid(which = "both", visible = False)
        axstore[i]["ax3"].grid(which = "both", visible = False)
            
            
    legend_items = [mpl.patches.Patch(facecolor = "blue", alpha = 0.5, hatch = "\\", lw = 2, label = f"{rad_threshold:.0%} Radiation")]
    fig.legend(handles = legend_items, loc = "lower right", bbox_to_anchor = (0.91,0.88), framealpha = 0)
    fig.subplots_adjust(wspace = 0)
    fig.suptitle(title)
    
    
        
def get_front_widths(p, s, Rcutoff = 0.7, dynamicGrid = True, use_prad = True):
    """
    Inputs
    -----
    p: dictionary containing the profile data (e.g. Btot)
    s: output store dictionary containing results (e.g. SpolPlot, crel)
    Rcutoff: define front to be where total cumulative radiation crosses this threshold
    
    
    """

    df = pd.DataFrame()

    for i, _ in enumerate(s["SpolPlot"]):
    
        
        if dynamicGrid is True:
            Spol = s["Spolprofiles"][i]
            S = s["Sprofiles"][i]
            Btot = s["Btotprofiles"][i]
            Bpol = s["Bpolprofiles"][i]
            Xpoint = s["Xpoints"][i]
        else:
            Spol = p["Spol"]
            S = p["S"]
            Btot = p["Btot"]
            Bpol = p["Bpol"]
            Xpoint = p["Xpoint"]
            
        q = pad_profile(S, s["Qprofiles"][i])  # W/m2
        T = pad_profile(S, s["Tprofiles"][i])
        R = pad_profile(S, s["Rprofiles"][i])
        
        Tu = s["Tprofiles"][i][-1]
        Nu = s["state"].nu
        Pu = Tu * s["state"].nu
        N = Pu / T
        Lfunc = s["constants"]["Lfunc"]
        Lz = np.array([Lfunc(x) for x in T])
        
        Rcum = sp.integrate.cumulative_trapezoid(y = R, x = S, initial = 0)
        Rcum /= Rcum.max()
        
        Beff = np.sqrt(sp.integrate.trapezoid(y = q*R, x = S) / sp.integrate.trapezoid(y = q*R/Btot**2, x = S))
        Beff_old = sp.integrate.trapezoid(y = R*Btot, x = S) / sp.integrate.trapezoid(y = R, x = S)
        
        
        if use_prad is True:
                

            ## Radiated power in W (includes B effects)
            Prad = np.gradient(q/Btot, S)
            
            if s["radios"]["upstreamGrid"] is True:
                # qradial is the uniform upstream heat source
                qradial = np.ones_like(S)
                qradial[Xpoint:] = s["state"].qradial
                Prad += qradial/Btot
            
            Pradcum = sp.integrate.cumulative_trapezoid(y = Prad, x = S, initial = 0)
            Pradcum /= np.nanmax(Pradcum)
            Rchoice = Pradcum
            if np.nanmax(Pradcum) == np.inf:
                raise Exception("Radiation is inf")

            
        else:
            Rcum = sp.integrate.cumulative_trapezoid(y = R, x = S, initial = 0)
            Rcum /= np.nanmax(Rcum)
            Rchoice = Rcum

        def get_front_ends(x):
            
            front_start_idx = np.argmin(abs(Rchoice - Rchoice[Rchoice> 0.01*Rchoice.max()][0]))
            front_end_idx = np.argmin(abs(Rchoice - Rcutoff))
            
            return x[front_start_idx], x[front_end_idx]
        
        
        
        df.loc[i, "Spol"] = s["SpolPlot"][i]
        df.loc[i, "Spol_front_end"], df.loc[i, "Spol_front_start"] = get_front_ends(Spol)
        df.loc[i, "Spol_front_width"] = df.loc[i, "Spol_front_start"] - df.loc[i, "Spol_front_end"]
        df.loc[i, "Spol_5eV"] = s["SpolPlot"][np.argmin(abs(T - 0.05))]
        
        df.loc[i, "Spar"] = s["Splot"][i]
        df.loc[i, "Spar_front_end"], df.loc[i, "Spar_front_start"] = get_front_ends(S)
        df.loc[i, "Spar_front_width"] = df.loc[i, "Spar_front_start"] - df.loc[i, "Spar_front_end"]
        df.loc[i, "Spar_5eV"] = s["Splot"][np.argmin(abs(T - 0.05))]

        if "cvar_trim" in s:
            suffix = "_trim"
        else:
            suffix = ""
            
        df.loc[i, "cvar"] = s[f"cvar{suffix}"][i]
        
        
        df.loc[i, "cvar"] = s[f"cvar{suffix}"][i]
        
        if type(s[f"crel{suffix}"]) == list:
            df.loc[i, "crel"] = s[f"crel{suffix}"][i]
            df.loc[i, "crel"] = s[f"crel{suffix}"][i]
        
        df.loc[i, "Btot"] = np.interp(df.loc[i, "Spar"], S, Btot)
        
        df.loc[i, "Btot_eff"] = Beff
        df.loc[i, "Btot_eff_old"] = Beff_old
        df.loc[i, "Bx"] = Btot[Xpoint]
        df.loc[i, "BxBt_eff"] = df.loc[i, "Bx"] / df.loc[i, "Btot_eff"]
    
        
        df.loc[i, "Tu"] = s["Tprofiles"][i][-1]
    
    return df

def get_detachment_scalings(profiles, stores, kappa0 = 2500):
    

    ## NOTE: Using power in W/m2 because we are using this calc to weigh Beff!
    front_dfs = [get_front_widths(profiles[x], stores[x], Rcutoff = 0.5, use_prad = True) for x in profiles]

    df = pd.DataFrame()
    df["thresholds"] = np.array([stores[x]["threshold"] for x in profiles])
    if "window_ratio" in list(stores.values())[0].keys():
        df["windows"] = np.array([stores[x]["window_ratio"] for x in profiles]) 
    df["L"] = np.array([profiles[x].get_connection_length() for x in profiles])
    df["BxBt"] = np.array([profiles[x].get_total_flux_expansion() for x in profiles])
    df["Bf"] = np.array([profiles[x]["Btot"][0] for x in profiles])
    df["frac_gradB"] = np.array([profiles[x].get_average_frac_gradB() for x in profiles])
    df["avgB_ratio"] = np.array([profiles[x].get_average_B_ratio() for x in profiles])
    df["BxBt_eff"] = [df["BxBt_eff"].iloc[0] for df in front_dfs]
    df["Btot_eff"] = [df["Btot_eff"].iloc[0] for df in front_dfs]
    df["Lx"] = [profiles[x]["S"][profiles[x]["Xpoint"]] for x in profiles]
    df["Tu"] = [stores[x]["Tprofiles"][0][-1] for x in profiles]
    df["Wradial"] = [stores[x]["Wradials"][0] for x in profiles]
    df["qx"] = [stores[x]["Qprofiles"][0][stores[x]["Xpoints"][0]] for x in profiles]

    ## Get effective avgB ratio
    avgB_ratio_eff = []
    for fdf in front_dfs:
        newS = np.linspace(fdf["Spar"].iloc[0], fdf["Spar"].iloc[-1], 100)
        # Btot_eff_interp = sp.interpolate.make_interp_spline(fdf["Spar"], fdf["Btot_eff"])(newS)
        Btot_eff_interp = np.interp(newS, fdf["Spar"], fdf["Btot_eff"])
        avgB_ratio_eff.append(fdf["Btot"].iloc[-1] / np.mean(Btot_eff_interp))
        
    df["avgB_ratio_eff"] = avgB_ratio_eff
    
    ### Get C0, classic DLS thresholds
    for i, key in enumerate(stores):
        store = stores[key]
        profile = profiles[key]
        S = store["Sprofiles"][0]   # Careful, this is the input profile not the profile the DLS ran with which is refined
        Lpar = S[-1]
        Xpoint = store["Xpoints"][0]
        Btot = store["Btotprofiles"][0]
        Sx = S[Xpoint]
        Bx = Btot[Xpoint]
        qpar = store["Qprofiles"][0]
        T = store["Tprofiles"][0]
        Tu = T[-1]
        Lfunc = store["constants"]["Lfunc"]
        Lz = [Lfunc(x) for x in T]
        T_arb = np.linspace(1, 300, 100)  # Arbitrary temp profile for the simple DLS cooling curve integral
        Lz_arb = [Lfunc(x) for x in T_arb]  
        Wradial = store["Wradials"][0]
        # Wradial_simple proportional to upstream q integral from DLS-simple
        Wradial_simple = sp.integrate.trapezoid(y = Btot[Xpoint:]/Bx * (Lpar - S[Xpoint:])/(Lpar - Sx), x = S[Xpoint:]) 
    
                
                
        df.loc[i, "Bx"] = Bx
        ### New derivation terms
        #   Kappa is hardcoded
        
        ## Impact of radiation happening upstream (no easy physical explanation)
        df.loc[i, "int_qoverBsq_dt"] = np.sqrt(2 * sp.integrate.trapz(y = qpar[Xpoint:]/(Btot[Xpoint:]**2 * Wradial), x = S[Xpoint:]))
        
        
        ## Tu proportional term calculated from heat flux integral. Includes effects of Lpar and B/averageB.
        #   Simple version is just the Tu proportionality
        df.loc[i, "W_Tu"] = (Wradial**(2/7)) / (sp.integrate.trapezoid(y = qpar, x = S)**(2/7))
        df.loc[i, "W_Tu_simple"] = (
                                        (sp.integrate.trapezoid(y = Btot[Xpoint:]/Bx * (Lpar - S[Xpoint:])/(Lpar - Sx), x = S[Xpoint:]) \
                                        + sp.integrate.trapezoid(y = Btot[:Xpoint]/Bx, x = S[:Xpoint]))
                                    )**(-2/7)                    
                                    
        ## Cooling curve integral which includes effect of Tu clipping integral limit
        df.loc[i, "int_TLz_dt"] = np.sqrt(2 * sp.integrate.trapz(y = 2500 * T**0.5 * Lz, x = T))**-1

        
        
        ####################################################################################
        ### Old derivation terms
        ####################################################################################
    
        ## C0

        ## Cooling curve integral
        # Effect of Tu clipping the integral in wide curves
        df.loc[i, "C0"] = 7**(-2/7) * (2*kappa0)**(-3/14) * (sp.integrate.trapezoid(y = Lz*T**0.5, x = T))**(-0.5)
        # df.loc[i, "C0"] = 7**(-2/7) * (2*kappa0)**(-3/14) * (sp.integrate.trapezoid(y = Lz*T**0.5 * Btot**(-2), x = T))**(-0.5)
    
        ## Classic DLS thresholds
        # df.loc[i, "DLS_thresholds"] = CfInt(
        #                                     spar = profile["S"], 
        #                                     B_field = profile["Btot"], 
        #                                     sx = profile["S"][profile["Xpoint"]], 
        #                                     L = profile["S"][-1],
        #                                     sh = store["Splot"][0]
        #                                 )
        
        ## Linear upstream integral
        # Accounts for the fact that qpar is comes in to the domain gradually upstream
        # Which means the L and Bx/Bt and other effects will also come in gradually
        abovex = sp.integrate.trapezoid(y = Btot[Xpoint:]/Bx * (Lpar - S[Xpoint:])/(Lpar - Sx), x = S[Xpoint:])
        belowx = sp.integrate.trapezoid(y = Btot[:Xpoint]/Bx, x = S[:Xpoint])
        df.loc[i, "upstream_integrals_linear"] = abovex + belowx 
        
        ## Upstream integral
        S = store["Sprofiles"][0]
        df.loc[i, "qpar_integral"] = sp.integrate.trapezoid(y = qpar, x = S)
        
        ## Cyd correction
        # Something to do with the upstream heat flux being gradually admitted and having an effect
        # that's distinct to the above one, and specifically about converting from a distance to a 
        # temperature integral that is no longer easy to do in DLS-Extended
        
        Xpoint = store["Xpoints"][0]
        S = store["Sprofiles"][0]
        Btot = store["Btotprofiles"][0]
        
        df.loc[i, "cyd_correction"] = np.sqrt(sp.integrate.trapz(y = qpar[Xpoint:]/(Btot[Xpoint:])**2, x = S[Xpoint:]) * store["state"].qradial)

        
        

    return df

def get_c0(store, Lfunc, kappa0 = 2500):
    idx = 0
    
    x = store["Sprofiles"][idx]
    T = pad_profile(x, store["Tprofiles"][idx])
    Tgrid = np.linspace(T.min(), T.max(), 1000)
    
    L = [Lfunc(x) for x in T]


    C0 = 7**(-2/7) * (2*kappa0)**(-3/14) * (sp.integrate.trapezoid(y = L, x = T))**0.5
    
    return C0

def pad_profile(S, data):
    """
    DLS terminates the domain at the front meaning downstream domain is ignored.
    This adds zeros to a result array data to fill those with zeros according to 
    the distance array S.
    """

    intended_length = len(S)
    actual_length = len(data)

    out = np.insert(data, 0, np.zeros((intended_length - actual_length)))
    
    return out


def get_sensitivity(crel_trim, SpolPlot, fluctuation=1.1, location=0, verbose = False):
        """
        Get detachment sensitivity at a certain location
        Sensitivity defined the location of front after a given fluctuation
        as a fraction of the total poloidal leg length.
        
        Inputs
        ------
        crel_trim: 1D array
            Crel values of detachment front with unstable regions trimmed (from DLS)
        SpolPlot: 1D array
            Poloidal distance from the DLS result
        fluctuation: float
            Fluctuation to calculate sensitivity as fraction of distance to X-point
            Default: 1.1
        location: float
            Location to calculate sensitivity as fraction of distance to X-point
            Default: target (0)
        verbose: bool
            Print results
            
        Returns
        -------
        sensitivity: float
            Sensitivity: position of front as fraction of distance towards X-point
            
        """
        # Drop NaNs for points in unstable region
        xy = pd.DataFrame()
        xy["crel"] = crel_trim
        xy["spol"] = SpolPlot
        xy = xy.dropna()

        Spol_from_crel = sp.interpolate.InterpolatedUnivariateSpline(xy["crel"], xy["spol"])
        Crel_from_spol = sp.interpolate.InterpolatedUnivariateSpline(xy["spol"], xy["crel"])

        Spol_at_loc = xy["spol"].iloc[-1] * location
        Crel_at_loc = Crel_from_spol(Spol_at_loc)
        Spol_total = xy["spol"].iloc[-1]


        if (Crel_at_loc - xy["crel"].iloc[0]) < -1e-6:
            sensitivity = 1   # Front in unstable region
        else:
            sensitivity = Spol_from_crel(Crel_at_loc*fluctuation) / Spol_total

        if verbose is True:
            print(f"Spol at location: {Spol_at_loc:.3f}")
            print(f"Crel at location: {Crel_at_loc:.3f}")
            print(f"Sensitivity: {sensitivity:.3f}")
            
        return sensitivity
    
    
    
def get_band_widths(d, o, cvar, size = 0.05):
    
    """ 
    Calculate detachment band widths based on geometry d and results o
    Size is the fraction of the band width to be calculated
    WORKS IN POLOIDAL
    """
    band_widths = {}
    # Find first valid index
    # This trims the unstable region on the inner
    # trim_idx = 0
    # for i,_ in enumerate(o["cvar_trim"]):
    #     if pd.isnull(o["cvar_trim"][i]) and not pd.isnull(o["cvar_trim"][i+1]):
    #         trim_idx = i+1
        
    # Make band based on topology dictionary (d) and results dictionary (o)
    # and a desired S poloidal location of the band centre (spol_middle)
    # as well as band size as a fraction (default +/-5%)
    trim_idx = 0
    crel = np.array(o["crel"])[trim_idx:]
    splot = np.array(o["Splot"])[trim_idx:]
    spolplot = np.array(o["SpolPlot"])[trim_idx:]

    if cvar == "power":
        crel = 1/crel

    c_grid = np.linspace(crel[0], crel[-1], 1000)
    k = 5
    spar_interp = sp.interpolate.interp1d(crel, splot, kind="cubic", fill_value = "extrapolate")
    spol_interp = sp.interpolate.interp1d(crel, spolplot, kind="cubic", fill_value = "extrapolate")

    band_widths = []
    for i, c_start in enumerate(c_grid):
        c_middle = c_start/(1-size)
        c_end = c_middle*(1+size)

        spar_start = spar_interp(c_start)
        spar_middle = spar_interp(c_middle)
        spar_end = spar_interp(c_end)

        spol_start = spol_interp(c_start)
        spol_middle = spol_interp(c_middle)
        spol_end = spol_interp(c_end)

        # band_widths.append(s_end - s_start)
        # band_widths.append(interp(c_start/(1-size)*(1+size)) - interp(c_start))
        band_width = spol_end - spol_start
        # band_width = spar_end - spar_start

        if band_width <= 0:
            band_widths.append(np.nan)
        elif spol_end > spolplot[-1]:
            band_widths.append(np.nan)
        else:
            band_widths.append(band_width)
            
    x = spol_interp(c_grid)
            
    return x, band_widths



def plot_morph_results(profiles, 
                       store, 
                       colors = ["teal", "darkorange"], 
                       xlabel = "Profile change",
                       show_47 = False,
                       sens = True):

    fig, axes = plt.subplots(2,3, dpi = 110, figsize = (12,8))

    
    thresholds = np.array([store[x]["threshold"] for x in profiles])
    windows = np.array([store[x]["window_ratio"] for x in profiles]) 
    L = np.array([profiles[x].get_connection_length() for x in profiles])
    BxBt = np.array([profiles[x].get_total_flux_expansion() for x in profiles])
    frac_gradB = np.array([profiles[x].get_average_frac_gradB() for x in profiles])
    avgB_ratio = np.array([profiles[x].get_average_B_ratio() for x in profiles])

    if sens:
        target_sens = np.array([get_sensitivity(store[x]["crel_trim"], store[x]["SpolPlot"], fluctuation=1.05, location=0.0) for x in profiles])

    L_base = L[0]
    BxBt_base = BxBt[0]
    threshold_base = thresholds[0]
    window_base = windows[0]
    avgB_ratio_base = avgB_ratio[0]
        
    threshcalc = (BxBt/BxBt_base)**(-1) * (L/L_base)**(-2/7) * (avgB_ratio/avgB_ratio_base)**(2/7)
    threshcalc2 = (BxBt/BxBt_base)**(-1) * (L/L_base)**(-4/7) * (avgB_ratio/avgB_ratio_base)**(2/7)

    windowcalc = (BxBt/BxBt_base)**(1) * (L/L_base)**(2/7) * (avgB_ratio/avgB_ratio_base)**(-2/7)
    windowcalc2 = (BxBt/BxBt_base)**(1) * (L/L_base)**(4/7) * (avgB_ratio/avgB_ratio_base)**(-2/7)

    index = list(profiles.keys())
    index /= index[0]

    # index = np.linspace(0,1,5)+i
    axes[0,0].set_title("Connection length")
    axes[0,0].plot(index, L/L_base, marker = "o")

    axes[0,1].set_title(r"Total flux expansion $\frac{B_{X}}{B_{t}}$")
    axes[0,1].plot(index, BxBt/BxBt_base, marker = "o")

    axes[0,2].set_title(r"Average $(\frac{1}{B}) \frac{dB}{ds_{pol}}$ below X-point")
    axes[0,2].plot(index, frac_gradB, marker = "o")

    axes[1,0].set_title(r"Detachment threshold")
    axes[1,0].plot(index, thresholds/threshold_base, marker = "o")

    if show_47:
        label_27 = "Analytical, $L^{2/7}$"
    else:
        label_27 = "Analytical"
    axes[1,0].plot(index, threshcalc, color = "darkslategrey", ls = "--", label = label_27, zorder = 100)
    
    if show_47:
        axes[1,0].plot(index, threshcalc2, color = "darkslategrey", ls = ":", label = "Analytical, $L^{4/7}$", zorder = 100)
    axes[1,0].legend(fontsize = "x-small")

    axes[1,1].set_title(r"Detachment window")
    axes[1,1].plot(index, windows/window_base, marker = "o")
    axes[1,1].plot(index, windowcalc, color = "darkslategrey", ls = "--", label = label_27, zorder = 100)
    
    if show_47:
        axes[1,1].plot(index, windowcalc2, color = "darkslategrey", ls = ":", label = "Analytical, $L^{4/7}$", zorder = 100)
    axes[1,1].legend(fontsize = "x-small")


    axes[1,2].set_title(r"10% Sensitivity @ target")
    
    if sens: 
        axes[1,2].plot(index, target_sens, marker = "o")

    for ax in axes.flatten():
        ax.set_xlabel(xlabel)

    for ax in [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]:
        ax.set_ylabel("Relative change")
        
    axes[0,2].set_ylabel(r"$(\frac{1}{B}) \frac{dB}{ds_{pol}}$")
    axes[1,2].set_ylabel(r"$S_{front, final} \ /\  S_{pol, x}$")
    fig.tight_layout()
    
    
def plot_profiles(base, stages, profiles, poloidal = True,  ylims = (None,None), reverse_S = True):

    fig, axes = plt.subplots(3,2, figsize = (10,10), dpi = 140)
    
    if poloidal is True:
        Schoice = "Spol"
    else:
        Schoice = "S"
        
    if reverse_S is True:
        plotx = lambda p: p[Schoice][-1] - p[Schoice][:p.Xpoint]
    else:
        plotx = lambda p: p[Schoice][:p.Xpoint] - p[Schoice][-1]

    ax = axes[0,0]
    ax.plot(base.R, base.Z, lw = 3, c = "k")
    
    for i, stage in enumerate(stages):

        if i > 0:
            kwargs = dict(markersettings = {"c" : "r"}, linesettings = {"c" : "r"})
        else:
            kwargs = {}
        stage.plot_control_points(ax = ax, **kwargs)
        
    # d_outer = eqb["SPR45"]["ol"]
    # ax.plot(d_outer["R"], d_outer["Z"], linewidth = 3, marker = "o", markersize = 0, color = "black", alpha = 1)
    ax.legend(fontsize = "small")
    if ylims != (None, None):
        ax.set_ylim(ylims)

    ax = axes[0,1]
    colors = [plt.cm.get_cmap("viridis", 5)(x) for x in range(5)]
    for idx, i in enumerate(profiles):
        p = profiles[i]
        ax.plot(p.R, p.Z, zorder = 100, alpha = 1, color = colors[idx], lw = 2, ls = "-")
        ax.scatter(p.R[0], p.Z[0], color= colors[idx], s = 10)
    ax.set_ylim(None, -5.5)
    ax.grid(alpha = 0.3, color = "k")
    ax.set_aspect("equal")
    if ylims != (None, None):
        ax.set_ylim(ylims)

    ax = axes[1,0]
    ax.set_title("Btot")
    for idx, i in enumerate(profiles):
        p = profiles[i]
        ax.plot(plotx(p), p.Btot[:p.Xpoint], zorder = 100, alpha = 1, color = colors[idx], lw = 2, ls = "-")
        
    ax = axes[1,1]
    # ax.set_title("Pitch")
    # for idx, i in enumerate(profiles):
    #     p = profiles[i]
    #     ax.plot((p.S[-1] - p.S), p.Bpol / p.Btot, zorder = 100, alpha = 1, color = colors[idx], lw = 2, ls = "-")
    
    ax.set_title("Fractional B par gradient")
    for idx, i in enumerate(profiles):
        p = profiles[i]
        ax.plot(plotx(p), np.gradient(p.Btot[:p.Xpoint], p.S[:p.Xpoint])/p.Btot[:p.Xpoint], zorder = 100, alpha = 1, color = colors[idx], lw = 2, ls = "-")
        
    ax = axes[2,0]
    L = [p.get_connection_length() for p in profiles.values()]
    L = L/L[0]
    BxBt = [p.get_total_flux_expansion() for p in profiles.values()]
    BxBt /= BxBt[0]
    # ax.plot((L/L[0]), zorder = 100, alpha = 1, color = colors[idx], lw = 0, marker = "o", ms = 10,  ls = "-")
    for idx, _ in enumerate(profiles):
        Lkwargs = dict(zorder = 100, alpha = 1, color = colors[idx], lw = 0, marker = "o", ms = 10,  ls = "-")
        BxBtkwargs = dict(zorder = 100, alpha = 1, color = colors[idx], lw = 0, marker = "x", ms = 10, markeredgewidth = 5, ls = "-")
        
        if idx == 0: Lkwargs["label"] = "Lc"
        if idx == 0: BxBtkwargs["label"] = r"$B_{X} \/ B_{t}$"
        ax.plot(idx, L[idx], **Lkwargs)
        ax.plot(idx, BxBt[idx], **BxBtkwargs)
        
    ax.legend()
        
    ax = axes[2,1]
    ax.set_title("Bpol")
    for idx, i in enumerate(profiles):
        p = profiles[i]
        ax.plot(plotx(p), p.Bpol[:p.Xpoint], zorder = 100, alpha = 1, color = colors[idx], lw = 2, ls = "-")
        
    ax.legend()
    fig.tight_layout()
    

def plot_results(
    stores, 
    mode,    # Inner or outer
    colors = None,
    ax = None, 
    dot_unstable = True,
    title = "DLS-Extended results"):
    
    spines = True
    
    if ax == None:
        fig, ax = plt.subplots(figsize = (6.5,6.5), dpi = 150)
        
    if len(stores) == 25:
        stores = {1 : stores}
        
    style = dict(lw = 2, ms = 0)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors == None else colors
    
    allx = []
    ally = []
    for i, id in enumerate(stores):
        s = stores[id]
        if dot_unstable is True:
            x = (s["cvar_trim"] / s["cvar_trim"][-1])
        else:
            x = (s["cvar"] / s["cvar"][-1])

        y = s["SpolPlot"] - s["SpolPlot"][-1]
        ax.plot(x, y, label = id, color = colors[i], **style, marker = "o")
        
        if i == 0: ax.scatter(x[-1], y[-1], color = "darkslategrey", marker = "x", linewidths = 4, s = 200, zorder = 10)
        ax.scatter(x[0], y[0], color = colors[i], marker = "o", linewidths = 4, s = 20, zorder = 10)
        
        if np.isnan(x).any():
            
            pointx = x[~np.isnan(x)][0]
            ax.plot([pointx, pointx], [y[np.where(x == pointx)[0][0]], y.min()], color = colors[i], linestyle = ":", linewidth = 2)
            ax.scatter(pointx, y.min(), color = colors[i], marker = "o", linewidths = 1, s = 25, zorder = 10)
            
        allx += list(x)
        ally += list(y)
        
    # ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(10))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
        
    # if max(allx) > 3:
    #     # Xaxis major, minor
    #     ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.4))
    #     ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.2))
    # else:
    #     ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    #     ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        
        
    # if max(ally) > 2:
    #     # Yaxis major, minor
    #     ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.4))
    #     ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.2))
    # else:
    #     # Yaxis major, minor
    #     ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    #     ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    
    def custom_formatter(x, pos):
        if x < 0:
            return f'{-x:.1f}'  # Multiply by -1 and format to 2 decimal places
        else:
            return f'{x:.1f}'
    
    ax.yaxis.set_major_formatter(custom_formatter)
    
    ax.grid(which = "major", visible = True,   alpha = 0.6)
    ax.grid(which = "minor", visible = True,  alpha = 0.3)
    
    if mode == "inner":
        ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel("Power increase factor", fontsize = "large")
    ax.set_ylabel("Poloidal distance from X-point $[m]$", fontsize = "large")
    ax.set_title(title, fontsize = "x-large", color = "black")
    
    if spines is True:
            for spine in ["top", "left", "right", "bottom"]:
                ax.spines[spine].set_visible(True)
                
                
    fig.show()
    
    
 
class plotProfiles():
    
    def __init__(self,
                 profiles,
                 colors = None,
                 basis = "Spol", 
                 side = "outer",
                 names = []):
    
        
        self.xlabel = dict(
                # Spol = r"$S_{\theta}\ [m]$",
                # S = r"$S_{\parallel}\ [m]$"
                Spol = r"Poloidal distance from X-point",
                S = r"Parallel distance from X-point"
            )[basis]
        
        self.side = side
        self.basis = basis
        self.profiles = profiles
        self.titlesize = "x-large"
        self.labelsize = "x-large"
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors == None else colors
        self.lw = 3
        self.figsize = (7,5)
        self.dpi = 150
        self.titlecolor = "black" #"#5D2785"
        self.spines = True
        
    def gridplot(self, field_list, eqbchoice, 
                 designs_xlims = (None,None),
                 designs_ylims = (None, None),
                 plot_all = False,
                 plot_simple = True,
                 profiles_only = False,
                 high_scale = False):
        figscale = 0.7
        if plot_all is True:
            fig, axes = plt.subplots(2,3, dpi = self.dpi, figsize = (18*figscale,9*figscale),
                            width_ratios = (3,2,2))
            
        elif profiles_only is True:
            if high_scale is True:
                scale = 1.8
            else:
                scale = 1
            fig, ax = plt.subplots(figsize = (8/scale,8/scale), dpi = 300*scale)
            
        elif plot_simple is True:
            figscale = 0.7
            fig = plt.figure(dpi = self.dpi, figsize = (14*figscale,11*figscale))
            shape = (4,6)
            # ax1 = plt.subplot2grid(shape = shape, loc = (0,0), colspan = 3, rowspan = 2)
            # ax2 = plt.subplot2grid(shape = shape, loc = (2,0), colspan = 3, rowspan = 2)
            # ax3 = plt.subplot2grid(shape = shape, loc = (1,3), colspan = 2, rowspan = 2)
            
            ax1 = plt.subplot2grid(shape = shape, loc = (0,0), colspan = 3, rowspan = 4)
            ax2 = plt.subplot2grid(shape = shape, loc = (0,3), colspan = 3, rowspan = 2)
            ax3 = plt.subplot2grid(shape = shape, loc = (2,3), colspan = 3, rowspan = 2)

            axes = [ax1, ax2, ax3]
            
        else:
            fig, axes = plt.subplots(2,2, dpi = self.dpi, figsize = (12*figscale,9*figscale),
                         width_ratios = (3,2))

        if profiles_only is True:
            self.plot_designs(eqbchoice, ax = ax, 
                            ylims = designs_ylims,
                            xlims = designs_xlims)
            ylims = ax.get_ylim()
            ax.vlines(0, *ylims, color = "k", ls = "--")
            ax.set_ylim(ylims)
            
        elif plot_simple is True:
            self.plot_designs(eqbchoice, ax = axes[0], 
                            ylims = designs_ylims,
                            xlims = designs_xlims)
            self.plot_Lc_BxBt(ax = axes[1])
            
        else:
            self.plot_designs(eqbchoice, ax = axes[0,0], 
                            ylims = designs_ylims,
                            xlims = designs_xlims)
            self.plot_Lc_BxBt(ax = axes[1,0])
        
        if profiles_only is True:
            pass
        elif plot_all is True:
            self.plot_field("Btot", ax = axes[0,1])
            self.plot_field("Bpol", ax = axes[1,1])
            self.plot_field("Btotgrad", ax = axes[0,2])
            self.plot_field("Bpitch", ax = axes[1,2])
        elif plot_simple is True:
            self.plot_field(field_list[1], ax = axes[2])
        else:
            self.plot_field(field_list[0], ax = axes[0,1])
            self.plot_field(field_list[1], ax = axes[1,1])

        fig.tight_layout()
        
        if plot_simple:
            fig.subplots_adjust(hspace = 0.9, wspace = 3.5)
        else:
            fig.subplots_adjust(hspace = 0.4)
        
    def plot_designs(self, eqbchoice, ax = [], 
                     xlims = (None, None),
                     ylims = (None, None)):
        """
        Plots profiles in R,Z space. Has both the inner and outer in the background, Needs eqb.
        The eqb dict must end with keys [il] and [ol] or a list of them
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize = self.figsize)
        else: 
            ax = ax

        for side in ["il", "ol"]:
            if type(eqbchoice) == list:
                for eqb in eqbchoice:
                    ax.plot(eqb[side]["R"], eqb[side]["Z"], c = "lightgrey", alpha = 1, lw = 3)
            else:
                ax.plot(eqbchoice[side]["R"], eqbchoice[side]["Z"], c = "lightgrey", alpha = 1, lw = 3)
            
        for i, key in enumerate(self.profiles.keys()):
            p = self.profiles[key]
            ax.plot(p["R"][:p["Xpoint"]], p["Z"][:p["Xpoint"]], color = self.colors[i], lw = self.lw)
            ax.scatter(p["R"][:p["Xpoint"]][0], p["Z"][:p["Xpoint"]][0], color = self.colors[i], marker = "o", linewidths = 4, s = 20, zorder = 10)
                    
        ax.set_aspect("equal")
        ax.yaxis.margin = 0
        ax.spines["bottom"].set_visible(False)
        ax.autoscale()

        if ylims != (None, None):
            ax.set_ylim(ylims)
        else:
            ax.set_ylim(None, -6.0)
            
        if xlims != (None, None):
            ax.set_xlim(xlims)
            
        ax.set_xlabel("R [m]", fontsize = self.labelsize)
        ax.set_ylabel("Z [m]", fontsize = self.labelsize)
        ax.set_title("Divertor configuration", fontsize = self.titlesize, color = self.titlecolor)
        
        if self.spines is True:
            for spine in ["top", "left", "right", "bottom"]:
                ax.spines[spine].set_visible(True)
        
    def plot_Lc_BxBt(self, ax = None, xlabels = False, figsize = None):
        """
        Plot summary of connection length and flux expansion
        """
        if ax == None:
            fig, ax = plt.subplots(figsize = self.figsize if figsize == None else figsize)
        else:
            ax = ax
            
        profiles = self.profiles
        colors = self.colors
        
        
        for i, key in enumerate(profiles.keys()):
            p = profiles[key]
            selector = slice(None,p.Xpoint)
            xplot = (p[self.basis] - p[self.basis][p.Xpoint])[selector]
            
            L = [p.get_connection_length() for p in profiles.values()]
            BxBt = [p.get_total_flux_expansion() for p in profiles.values()]
            gradB = [p.get_gradB_integral() for p in profiles.values()]
            Bpitch = [p.get_Bpitch_integral() for p in profiles.values()]
            
            L = L/L[0]
            BxBt /= BxBt[0]
            gradB /= gradB[0]
            Bpitch /= Bpitch[0]
            
            Lkwargs = dict(zorder = 100, alpha = 1, color = colors[i], lw = 0, marker = "o", ms = 10,  ls = "-")
            BxBtkwargs = dict(zorder = 100, alpha = 1, color = colors[i], lw = 0, marker = "x", ms = 10, markeredgewidth = 5, ls = "-")
            
            if i == 0: Lkwargs["label"] = "Connection length factor"
            if i == 0: BxBtkwargs["label"] = "Flux expansion factor"
            # ax.plot(i, L[i], **Lkwargs)
            # ax.plot(i, BxBt[i], **BxBtkwargs)
            
            ax.plot(i, gradB[i], marker = "o", c = "k")
            
        # ax.set_title("Flux expansion and connection length", fontsize = self.titlesize, color = self.titlecolor)
        ax.set_ylabel("Factor", fontsize = self.labelsize)
        ax.set_xlabel("Profile change", fontsize = self.labelsize)
        # leg = ax.legend(loc = "upper left", bbox_to_anchor = (0.02,0.95), fontsize = "medium")
        leg = ax.legend(fontsize = "medium", frameon = False)
        leg.get_frame().set_linewidth(0)
        
        
        if xlabels is True:
            ax.set_xticks(range(len(profiles)))
            ax.set_xticklabels(self.profiles.keys())
        else:
            ax.set_xticklabels([])
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis = "x", which = "both", length = 0)
            ax.grid(axis = "x", which = "both", alpha = 0)
        
        ax.set_ylabel("Factor", fontsize = self.labelsize)
        if self.spines is True:
            for spine in ["top", "left", "right", "bottom"]:
                ax.spines[spine].set_visible(True)
        else:
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(False)
            
        ax.set_title("Profile properties", fontsize = self.titlesize, color = self.titlecolor)
        
    def plot_summary(self, ax = None, xlabels = False, figsize = None):
        """
        Plot summary of connection length and flux expansion
        """
        if ax == None:
            fig, ax = plt.subplots(figsize = self.figsize if figsize == None else figsize)
        else:
            ax = ax
            
        profiles = self.profiles
        colors = self.colors
        
        
        L = [p.get_connection_length() for p in profiles.values()]
        BxBt = [p.get_total_flux_expansion() for p in profiles.values()]
        gradB = [p.get_gradB_average() for p in profiles.values()]
        Bpitch = [p.get_Bpitch_average() for p in profiles.values()]
        
        L = L/L[0]
        BxBt /= BxBt[0]
        gradB /= gradB[0]
        Bpitch /= Bpitch[0]
        
        # Lkwargs = dict(zorder = 100, alpha = 1, color = "darkorange", lw = 0, marker = "o", ms = 10,  ls = "-")
        # BxBtkwargs = dict(zorder = 100, alpha = 1, color = "purple", lw = 0, marker = "x", ms = 10, markeredgewidth = 5, ls = "-")
        
        common_kwargs = dict(zorder = 100, alpha = 1, lw = 3, ms = 5)
        
        x = range(len(profiles))
        
        ax.plot(x, L, label = "Connection length factor", marker = "o", c = "darkorange", **common_kwargs)
        ax.plot(x, BxBt, label = "Flux expansion factor", marker = "o", c = "purple", **common_kwargs)
        
        ax.plot(x, gradB, label = "Average B gradient", ls = "--", c = "forestgreen", **common_kwargs)
        ax.plot(x, Bpitch, label = "Average of B pitch", ls = "--", c = "deeppink", **common_kwargs)
            
        # ax.set_title("Flux expansion and connection length", fontsize = self.titlesize, color = self.titlecolor)
        ax.set_ylabel("Factor", fontsize = self.labelsize)
        ax.set_xlabel("Profile change", fontsize = self.labelsize)
        # leg = ax.legend(loc = "upper left", bbox_to_anchor = (0.02,0.95), fontsize = "medium")
        leg = ax.legend(fontsize = "medium", frameon = False)
        leg.get_frame().set_linewidth(0)
        
        
        if xlabels is True:
            ax.set_xticks(range(len(profiles)))
            ax.set_xticklabels(self.profiles.keys())
        else:
            ax.set_xticklabels([])
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis = "x", which = "both", length = 0)
            ax.grid(axis = "x", which = "both", alpha = 0)
        
        ax.set_ylabel("Factor", fontsize = self.labelsize)
        if self.spines is True:
            for spine in ["top", "left", "right", "bottom"]:
                ax.spines[spine].set_visible(True)
        else:
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(False)
            
        ax.set_title("Profile properties", fontsize = self.titlesize, color = self.titlecolor)
        
    def plot_field(self, mode, ax = None, figsize = None):
        """
        Plot profiles of either the field, field gradient or pitch
        """
        
        if ax == None:
            fig, ax = plt.subplots(figsize = self.figsize if figsize == None else figsize)
        else:
            ax = ax
            
        profiles = self.profiles
        colors = self.colors
        
        data = dict(
                Btot = {"y" : lambda x: x.Btot, 
                        "ylabel" : r"$B_{tot}\ [T]$",
                        "title" : r"Total field"},
                
                Bpol = {"y" : lambda x: x.Bpol, 
                        "ylabel" : r"$B_{\theta}\ [T]$",
                        "title" : "Poloidal field"},
                
                Btotgrad = {"y" : lambda x: (np.gradient(x.Btot, x.Spol)/x.Btot), 
                            "ylabel" : r"$\frac{1}{B}\ \frac{dB}{dS_{\theta}}\ [m^{-1}]$",
                            "title" : "B field gradient"},
                
                Bpitch = {"y" : lambda x: (x.Bpol/x.Btot), 
                          "ylabel" : r"$B_{\theta}\ /\ B_{tot}$",
                          "title" : "B field pitch"}
            )
        
        for i, key in enumerate(profiles.keys()):
            
            p = profiles[key]
            selector = slice(None,p.Xpoint)
            xplot = (p[self.basis] - p[self.basis][p.Xpoint])[selector] * -1
            

            ax.plot(xplot, data[mode]["y"](p)[selector], label = key, color = colors[i], lw = self.lw)
            ax.scatter(xplot[0], data[mode]["y"](p)[selector][0], color = colors[i], marker = "o", linewidths = 4, s = 20, zorder = 10)
            ax.scatter(xplot[-1], data[mode]["y"](p)[selector][-1], color = colors[i], marker = "x", linewidths = 3, s = 100, zorder = 10)
            ax.set_title(data[mode]["title"], fontsize = self.titlesize, color = self.titlecolor)
            ax.set_xlabel(self.xlabel, fontsize = self.labelsize)
            ax.set_ylabel(data[mode]["ylabel"], fontsize = self.labelsize)
            
        if self.side == "inner":
            ax.set_xlim(ax.get_xlim()[::-1])
            
        if self.spines is True:
            for spine in ["top", "left", "right", "bottom"]:
                ax.spines[spine].set_visible(True)
        
    


class compare_SOLPS_DLS():
    def __init__(self, slc, out, cvar, impurity = "Ar", region = "outer_lower", sepadd = 1):

        # Read SOLPS
        self.solps = slc.get_1d_poloidal_data(["Btot", "hx", "vol", "R", 
                                               "Te", "Td+", f"R{impurity}", f"f{impurity}", "ne", "fhex_cond", "fhx_total",
                                               ], 
                                  sepadd = sepadd, region = region, target_first = True)
        
        self.impurity = impurity
        
        
        DLScase = FrontLocation(out)
        self.dls = DLScase.data
        self.dls["qpar_cond_frac"] = 1
        self.dls["Ne_sq_cz"] = self.dls["Ne"]**2 * self.dls["cz"]
        
        self.solps = self.calculate_solps(self.solps)
        
        # # Read DLS
        # dls = pd.DataFrame()
        # dls["Qrad"] = out["Qrad_profiles"][0]
        # dls["Spar"] = out["Spar_profiles"][0]
        # dls["Spol"] = out["Spol_profiles"][0]
        # dls["Te"] = out["Te_profiles"][0]
        # dls["qpar"] = out["qpar_profiles"][0]
        # dls["Btot"] = out["Btot_profiles"][0]
        # dls["qpar_over_B"] = dls["qpar"] / dls["Btot"]
        
        # if cvar == "density":
        #     dls["cz"] = out.inputs.cz0
        #     dls["Ne"] = out["cvar"][0] * dls["Te"].iloc[-1] / dls["Te"]   ## Assuming cvar is ne
        # elif cvar == "impurity_frac":
        #     dls["cz"] = out["cvar"][0]
        #     dls["Ne"] = dls.iloc[-1]["Te"] * out.inputs.nu0 / dls["Te"]
        # else:
        #     raise Exception()
        
        # Xpoint = out["Xpoints"][0]
        # dls.loc[Xpoint, "Xpoint"] = 1

        # # qradial is the uniform upstream heat source
        # dls["qradial"] = 1.0
        # # dls["qradial"].iloc[Xpoint:] = out["state"].qradial
        # dls.loc[Xpoint:, "qradial"] = out["state"].qradial
        
        # self.dls = dls
        
        
        
        
        # self.dls = self.calculate_dls(self.dls)
        
        
    def calculate_solps(self, solps):
        
        for param in solps:
            if param.startswith("fh"):
                solps[param] /= solps["apar"]  # Parallel cross-sectional area


        # Calculate radiation (see DLS comments)
        solps["Qrad_per_area_cum"] = sp.integrate.cumulative_trapezoid(y = solps[f"R{self.impurity}"], x = solps["Spar"], initial = 0)   # Radiation integral over volume
        solps["Qrad_per_area_cum_norm"] = solps["Qrad_per_area_cum"] / solps["Qrad_per_area_cum"].max()

        solps["Qrad_cum"] = sp.integrate.cumulative_trapezoid(y = solps[f"R{self.impurity}"] / solps["Btot"], x = solps["Spar"], initial = 0)   # Radiation integral over volume
        # NOTE: This is not correct, it should really be volume, but I'm doing  it like this to be the same as the DLS.
        solps["Qrad_cum_norm"] = solps["Qrad_cum"] / solps["Qrad_cum"].max()
        
        solps["Pe"] = solps["Te"] * solps["ne"] * 1.60217662e-19
        solps["Ne"] = solps["ne"]
        
        solps["fhex_cond"] = solps["fhex_cond"].abs()
        solps["fhx_total"] = solps["fhx_total"].abs()
        solps["qpar"] = solps["fhx_total"]
        solps["qpar_cond"] = solps["fhex_cond"]
        
        cond_capped = np.minimum(solps["fhex_cond"], solps["fhx_total"])
        
        ratio = cond_capped / solps["fhx_total"]
        ratio.iloc[-2] = np.mean([ratio.iloc[-3], ratio.iloc[-1]])
        solps["qpar_cond_frac"] = ratio
        
        solps["Ne_sq_cz"] = solps["Ne"]**2 * solps[f"f{self.impurity}"]
        
                
        return solps

    # def calculate_dls(self, dls):
        
        
    #     # Radiative power loss without flux expansion effect.
    #     # Units are W, bit integrated assuming unity cross-sectional area, so really W/m2
    #     # Done by reconstructing the RHS of the qpar equation
    #     dls["Prad_per_area"] = np.gradient(dls["qpar"]/dls["Btot"], dls["Spar"]) + dls["qradial"]/dls["Btot"]
    #     dls["Prad_per_area_cum"] = sp.integrate.cumulative_trapezoid(y = dls["Prad_per_area"], x = dls["Spar"], initial = 0)  # W/m2
    #     dls["Prad_per_area_cum_norm"] = dls["Prad_per_area_cum"] / dls["Prad_per_area_cum"].max()
    #     # Proper radiative power integral [W]
    #     dls["Prad_cum"] = sp.integrate.cumulative_trapezoid(y = dls["Qrad"] / dls["Btot"], x = dls["Spar"], initial = 0)   # Radiation integral over volume
    #     dls["Prad_cum_norm"] = dls["Prad_cum"] / dls["Prad_cum"].max()
        
    #     dls["Pe"] = dls["Te"] * dls["Ne"] * 1.60217662e-19
    #     dls["Ne_sq_cz"] = dls["Ne"]**2 * dls["cz"]
    #     dls["qpar_cond_frac"] = 1
        
    #     return dls
    
    def get_front_location(self, mode = "dls", threshold = 0.5, debug_plot = False):
        """
        Returns the parallel position by which the total cumulative radiation fraction reached a threshold
        """
        
        if mode == "dls":
            data = self.dls
        elif mode == "solps":
            data = self.solps
            
        df = data[["Qrad_cum_norm", "Spar"]].copy()
        df = df.drop_duplicates(subset = ["Qrad_cum_norm"])
        fun = sp.interpolate.interp1d(df["Qrad_cum_norm"], df["Spar"],  kind = "linear")
        
        if debug_plot:
            fig, ax = plt.subplots()
            ax.plot(df["Qrad_cum_norm"], df["Spar"], marker = "o", lw = 0, ms = 3)
            rad = np.linspace(0,1,100)
            ax.plot(rad, [fun(r) for r in rad])
        
        return fun(threshold)
    
    def plot(self, list_plots, 
             normalise_radiation = True, 
             radiation_per_area = False,
             plot_cz = False,
             legend_loc = "upper left",
             title = ""):
        
        self.legend_loc = legend_loc
        fsize = 4
        nfigs = len(list_plots)
        fig, axes =plt.subplots(1,nfigs, figsize = (nfigs*fsize, fsize))
        if title != "":
            fig.suptitle(title)
        
        for i, plot in enumerate(list_plots):
            if plot == "Ne":
                self.plot_Ne(axes[i])
            elif plot == "Te":
                self.plot_Te(axes[i])
            elif plot == "Pe":
                self.plot_Pe(axes[i])
            elif plot == "qpar":
                self.plot_qpar(axes[i])
            elif plot == "qpar_over_B":
                self.plot_qpar_over_B(axes[i])
            elif plot == "Qrad":
                self.plot_radiation_source(axes[i], logscale = True)
            elif plot == "Cumrad":
                self.plot_radiation_integral(axes[i], normalise = normalise_radiation, per_area= radiation_per_area, plot_cz = plot_cz)
            elif plot == "cz":
                self.plot_cz(axes[i])
            elif plot == "Ne_cz":
                self.plot_ne_cz(axes[i])
            else:
                print(f"Plot {plot} not found")
        
        fig.tight_layout()
        
    def plot_Xpoint(self, ax):
        # solps = self.solps
        dls = self.dls
        
        ylim = ax.get_ylim()
        ax.vlines(dls[dls["Xpoint"]==1]["Spar"], *ylim, colors = "k", alpha = 0.3, lw = 1, ls = "--", label = "X-point")
        ax.set_ylim(*ylim)
        
    def apply_plot_settings(self, ax):
        ax.legend(fontsize = "x-small", loc = self.legend_loc)
        ax.set_xlabel(r"$s_{\parallel}$ [m]")
        
    def plot_qpar(self, ax):
        
        solps = self.solps
        dls = self.dls
        
        ax.set_title("Parallel heat flux")
        ax.plot(dls["Spar"], dls["qpar"], label = "DLS")
        ax.plot(solps["Spar"], abs(solps["fhex_cond"]), label = "SOLPS electron cond")
        ax.plot(solps["Spar"], abs(solps["fhx_total"]), label = "SOLPS total")
        
        ax.set_ylabel(r"$q_{\parallel}$ $[Wm^{-2}]$")
        self.plot_Xpoint(ax)
        self.apply_plot_settings(ax)
        
    def plot_qpar_over_B(self, ax):
        
        solps = self.solps
        dls = self.dls
        
        ax.set_title("Parallel heat flux / $B_{tot}$")
        ax.plot(dls["Spar"], dls["qpar"]/dls["Btot"], label = "DLS")
        ax.plot(solps["Spar"], abs(solps["fhex_cond"])/solps["Btot"], label = "SOLPS electron cond")
        ax.plot(solps["Spar"], abs(solps["fhx_total"])/solps["Btot"], label = "SOLPS total")
        
        ax.set_ylabel(r"$q_{\parallel}/B_{tot}$ $[Wm^{-2}T^{-1}]$")
        self.plot_Xpoint(ax)
        self.apply_plot_settings(ax)
        
    def plot_Te(self, ax):
        
        solps = self.solps
        dls = self.dls
        
        ax.set_title("Temperature")
        ax.plot(dls["Spar"], dls["Te"], label = "DLS")
        ax.plot(solps["Spar"], solps["Te"], label = "SOLPS")
        ax.set_ylabel("$T_e$ $[eV]$")
        self.plot_Xpoint(ax)
        self.apply_plot_settings(ax)

    def plot_Ne(self, ax):
        solps = self.solps
        dls = self.dls
        
        ax.set_title("Density")
        ax.plot(dls["Spar"], dls["Ne"], label = "DLS")
        ax.plot(solps["Spar"], solps["ne"], label = "SOLPS")

        ax.set_ylabel("$N_e$ $[m^{-3}]$")
        ax.set_yscale("log")
        self.plot_Xpoint(ax)
        self.apply_plot_settings(ax)
        
    def plot_Pe(self, ax):
        solps = self.solps
        dls = self.dls
        
        ax.set_title("Electron pressure (normalised)")
        ax.plot(dls["Spar"], dls["Pe"] / dls["Pe"].iloc[-1], label = "DLS")
        ax.plot(solps["Spar"], solps["Pe"] / solps["Pe"].iloc[-1], label = "SOLPS")

        ax.set_ylabel("$P_e$ (normalised)")
        # ax.set_yscale("log")
        self.plot_Xpoint(ax)
        self.apply_plot_settings(ax)
        
    def plot_cz(self, ax):
        solps = self.solps
        dls = self.dls
        
        ax.set_title("fAr (normalised)")
        
        dls_var = dls["cz"]
        solps_var = solps[f"f{self.impurity}"]
        
        ax.plot(dls["Spar"], dls_var/dls_var.iloc[-1], label = "DLS")
        ax.plot(solps["Spar"], solps_var/solps_var.iloc[-1],  label = "SOLPS")

        ax.set_ylabel(f"f{self.impurity}")
        self.plot_Xpoint(ax)
        self.apply_plot_settings(ax)
        
    def plot_ne_cz(self, ax):
        solps = self.solps
        dls = self.dls
        
        ax.set_title("fAr * $n_e^2$ (normalised)")
        
        dls_var = dls["cz"]*dls["Ne"]**2
        solps_var = solps[f"f{self.impurity}"]*solps["ne"]**2
        
        ax.plot(dls["Spar"], dls_var/dls_var.iloc[-1], label = "DLS")
        ax.plot(solps["Spar"], solps_var/solps_var.iloc[-1],  label = "SOLPS")

        ax.set_ylabel(f"f{self.impurity} * $n_e^2$ [m^-6")
        ax.set_yscale("log")
        self.plot_Xpoint(ax)
        self.apply_plot_settings(ax)
        
    def plot_radiation_source(self, ax, logscale = True):
        
        solps = self.solps
        dls = self.dls
        
        ax.plot(dls["Spar"], dls["Qrad"], label = "DLS")
        ax.plot(solps["Spar"], solps[f"R{self.impurity}"], label = "SOLPS")
        ax.set_title("Radiation source")
        ax.set_ylabel("[W/m3]")

        if logscale:
            ax.set_yscale("log")
            
        self.plot_Xpoint(ax)
        self.apply_plot_settings(ax)
            
    def plot_radiation_integral(
        self, ax, 
        per_area = False, 
        logscale = False, 
        normalise = True,
        plot_cz = True):

        solps = self.solps
        dls = self.dls
        
        if normalise:
            suffix = "_norm"
            label_suffix = "(normalised)"
        else:
            suffix = ""
            label_suffix = ""
        
        if per_area:
            ax.plot(dls["Spar"], dls[f"Qrad_per_area_cum{suffix}"], label = "DLS")
            ax.plot(solps["Spar"], solps[f"Qrad_per_area_cum{suffix}"], label = "SOLPS")
            ax.set_title(f"Cumulative radiation\nintegral (per area) {label_suffix}")
            ax.set_ylabel("[W/m2]")
        else:
            ax.plot(dls["Spar"], dls[f"Qrad_cum{suffix}"], label = "DLS")
            ax.plot(solps["Spar"], solps[f"Qrad_cum{suffix}"], label = "SOLPS")
            ax.set_title(f"Cumulative radiation integral {label_suffix}")
            ax.set_ylabel(f"[W] {label_suffix}")
        
        if plot_cz:
            ax2 = ax.twinx()
            ax2.plot(solps["Spar"], solps[f"f{self.impurity}"] * solps["ne"]**2, 
                     c = "orange", ls = "--", alpha = 0.8, lw = 2, 
                     label = "fAr*ne^2")
            ax2.set_ylabel("Cz*ne**2")
            ax2.set_yscale("log")
            ax2.legend(fontsize = "x-small")
        
        if logscale:
            ax.set_yscale("log")
            
        self.plot_Xpoint(ax)
        self.apply_plot_settings(ax)
        
        
# class DLScase():
#     def __init__(self, out, index=0):

#         dls = pd.DataFrame()
#         dls["Qrad"] = out["Rprofiles"][index]
#         dls["Spar"] = out["Sprofiles"][index]
#         dls["Spol"] = out["Spolprofiles"][index]
#         dls["Te"] = out["Tprofiles"][index]
#         dls["qpar"] = out["Qprofiles"][index]
#         dls["Btot"] = out["Btotprofiles"][index]
#         dls["Ne"] = out["cvar"][index] * dls["Te"].iloc[-1] / dls["Te"]   ## Assuming cvar is ne
#         dls["cz"] = out["state"].si.cz0
#         Xpoint = out["Xpoints"][index]
#         dls.loc[Xpoint, "Xpoint"] = 1

#         # qradial is the uniform upstream heat source
#         dls["qradial"] = 1.0
#         # dls["qradial"].iloc[Xpoint:] = out["state"].qradial
#         dls.loc[Xpoint:, "qradial"] = out["state"].qradial
        
#         # Radiative power loss without flux expansion effect.
#         # Units are W, bit integrated assuming unity cross-sectional area, so really W/m2
#         # Done by reconstructing the RHS of the qpar equation
#         dls["Prad_per_area"] = np.gradient(dls["qpar"]/dls["Btot"], dls["Spar"]) + dls["qradial"]/dls["Btot"]
#         dls["Prad_per_area_cum"] = sp.integrate.cumulative_trapezoid(y = dls["Prad_per_area"], x = dls["Spar"], initial = 0)  # W/m2
#         dls["Prad_per_area_cum_norm"] = dls["Prad_per_area_cum"] / dls["Prad_per_area_cum"].max()
#         # Proper radiative power integral [W]
#         dls["Prad_cum"] = sp.integrate.cumulative_trapezoid(y = dls["Qrad"] / dls["Btot"], x = dls["Spar"], initial = 0)   # Radiation integral over volume
#         dls["Prad_cum_norm"] = dls["Prad_cum"] / dls["Prad_cum"].max()
        
#         dls["Pe"] = dls["Te"] * dls["Ne"] * 1.60217662e-19
#         dls["qpar_over_B"] = dls["qpar"] / dls["Btot"]
        
#         ### Calculate scalar properties
#         s = dict()
#         s["cvar"] = out["state"].cvar
#         s["kappa0"] = out["state"].si.kappa0
#         s["Bf"] = dls["Btot"].iloc[0]
#         s["Bx"] = dls[dls["Xpoint"]==1]["Btot"].iloc[0]
#         s["Beff"] = np.sqrt(sp.integrate.trapezoid(
#             y = dls["qpar"]*dls["Qrad"], x = dls["Spar"]) / sp.integrate.trapezoid(
#                 y = dls["qpar"]*dls["Qrad"]/dls["Btot"]**2, x = dls["Spar"]))
#         s["BxBt"] = s["Bx"] / s["Bf"]
#         s["BxBteff"] = s["Bx"] / s["Beff"]
#         s["Lc"] = dls["Spol"].iloc[-1]
#         s["Wradial"] = out["state"].qradial
#         s["Tu"] = dls["Te"].iloc[-1]
        
#         dlsx = dls[dls["Xpoint"] == 1]
#         dls_div = dls[dls["Spar"] <= dlsx["Spar"].iloc[0]]
#         avgB_div = sp.integrate.trapz(dls_div["Btot"], x = dls_div["Spar"]) / dls_div["Spar"].iloc[-1]

#         s["avgB_ratio"] = dlsx["Btot"].iloc[0] / avgB_div
        
#         # print(s["avgB_ratio"])
        
#         ## DLS-Extended effects (see Kryjak 2024)
#         # Impact of qpar profile changing upstream due to B field and radiation,
#         # leading to a different qpar at the X-point
#         s["upstream_rad"] = np.sqrt(2 * sp.integrate.trapz(
#             y = dls["qpar"].iloc[Xpoint:]/(dls["Btot"].iloc[Xpoint:]**2 * s["Wradial"]), x = dls["Spar"].iloc[Xpoint:]))
        
        
#         # Tu proportional term calculated from heat flux integral. Includes effects of Lpar and B/averageB.
#         # Simple version is just the Tu proportionality
#         s["W_Tu"] = (s["Wradial"]**(2/7)) / (sp.integrate.trapezoid(y = dls["qpar"], x = dls["Spar"])**(2/7))
#         s["W_Tu_simple"] = (
#                             (sp.integrate.trapezoid(
#                                 y = dls["Btot"][Xpoint:]/s["Bx"] * (s["Lc"] - dls["Spar"].iloc[Xpoint:])/(s["Lc"] - dls["Spar"].iloc[Xpoint]), x = dls["Spar"].iloc[Xpoint:]) \
#                             + sp.integrate.trapezoid(
#                                 y = dls["Btot"].iloc[:Xpoint]/s["Bx"], x = dls["Spar"].iloc[:Xpoint]))
#                         )**(-2/7)                    
                                    
#         # Cooling curve integral which includes effect of Tu clipping integral limit
#         self.Lfunc = lambda x : out["state"].si.Lfunc(x)
#         Lz = [self.Lfunc(x) for x in dls["Te"]]
#         s["curveclip"] = np.sqrt(2 * sp.integrate.trapz(y = s["kappa0"] * dls["Te"]**0.5 * Lz, x = dls["Te"]))**-1
        
#         self.data = dls
#         self.stats = s
        
        
#     def get_stats_dataframe(self):
#         """
#         Returns dataframe with a row of index 0 containing properties in self.stats
#         """
#         return pd.DataFrame.from_dict(self.stats, orient = "index").T
        
#     def get_radiation_location(self, threshold):
#         """
#         Returns parallel location of point by which the cumulative radiation in W (not W/m2) crosses a threshold
        
#         TRY 16 to 84
#         """
#         return sp.interpolate.interp1d(self.data["Prad_cum_norm"], self.data["Spar"], fill_value = "extrapolate")(threshold)
    
#     def get_front_width(self, lower_threshold, upper_threshold):
#         """
#         Return parallel difference between two cumulative radiation thresholds
#         Cumulative radiation is W not W/m2
#         """
        
#         lower_position = sp.interpolate.interp1d(self.data["Prad_cum_norm"], self.data["Spar"], fill_value = "extrapolate")(lower_threshold)
#         upper_position = sp.interpolate.interp1d(self.data["Prad_cum_norm"], self.data["Spar"], fill_value = "extrapolate")(upper_threshold)
#         width = upper_position - lower_position
        
#         return width
    
    
# class DLScasedeck:
#     def __init__(self, store):
#         num_locations = len(store["Sprofiles"])
#         self.cases = []
#         for i in range(num_locations):
#             self.cases.append(DLScase(store, index=i))

#         self.data = pd.DataFrame()
#         self.data["Spar"] = store["Spar_front"]
#         self.data["Spol"] = store["Spol_front"]
#         self.data["cvar"] = store["cvar"]   # cvar at detachment threshold
#         self.data["crel"] = store["cvar"] / store["cvar"][0]
        
#         self.single_case = "window" not in store.keys()
        
#         if self.single_case:
#             print("Warning, deck contains only one case! Detachment window and unstable region not available.")
#             self.window = 0
#             self.window_frac = 0
#             self.window_ratio = 0
#         else:
#             self.window = store["window"]             # Cx - Ct
#             self.window_frac = store["window_frac"]   # (Cx - Ct) / Ct
#             self.window_ratio = store["window_ratio"] # Cx / Ct
            
        
#         if len(self.data) != len(self.data.drop_duplicates(subset = "Spar")):
#             print("Warning: Duplicate Spar values found, removing!")
#             self.data = self.data.drop_duplicates(subset = "Spar")
        
#         if not self.single_case:
#             self.get_stable_region()
#         # self.get_stability_breakpoint()
        
        
#     def get_stable_region(self, diagnostic_plot = False):
#         """
#         Trim the crel and cvar arrays to only include stable region
#         """
        
#         self.data.loc[:, "crel_grad"] = np.gradient(self.data["crel"])
#         self.data.loc[:, "stable"] = False
#         self.data.loc[self.data["crel_grad"] > 0, "stable"] = True
#         data_stable = self.data[self.data["stable"] == True]
#         data_unstable = self.data[self.data["stable"] == False]

#         ## Size of unstable region when going backward
#         if len(data_stable) == 0:
#             self.unstable_Lpol_backward = self.data.iloc[-1]["Spol"]
#             self.unstable_Lpar_backward = self.data.iloc[-1]["Spar"]
#         elif len(data_unstable) > 0:
#             self.unstable_Lpol_backward = data_stable.iloc[0]["Spol"]
#             self.unstable_Lpar_backward = data_stable.iloc[0]["Spar"]
#         else:
#             self.unstable_Lpol_backward = 0
#             self.unstable_Lpar_backward = 0

#         if self.data["crel"].iloc[-1] < 1:
#             self.unstable_Lpol_forward = self.data["Spol"].iloc[-1]
#             self.unstable_Lpar_forward = self.data["Spar"].iloc[-1]
#         else:
#             self.unstable_Lpol_forward = sp.interpolate.interp1d(data_stable["crel"], data_stable["Spol"], kind = "linear")(1)
#             self.unstable_Lpar_forward = sp.interpolate.interp1d(data_stable["crel"], data_stable["Spar"], kind = "linear")(1)
            
#         if diagnostic_plot:
#             fig, ax = plt.subplots(dpi = 120)
#             ax.plot(self.data["crel"], self.data["Spol"], label = "Crel")
#             ax.plot(data_stable["crel"], data_stable["Spol"])
#             ax.hlines(self.unstable_Lpol_backward, self.data["crel"].min(), self.data["crel"].max(), ls = "--", color = "red", label = "forward stability breakpoint")
#             ax.hlines(self.unstable_Lpol_forward, self.data["crel"].min(), self.data["crel"].max(), ls = "--", color = "orange", label = "backward stability breakpoint")
#             ax.legend()
        
#     def get_stability_breakpoint(self):
#         """
#         Find the exact point in Crel where the stable region starts
#         through interpolation.
#         Forward and backward refer to the direction of front movement.
#         Unstable regions see hysteresis where the stable region going towards the
#         target is greater.
#         """
        
#         crel = self.data["crel"]
#         Spar = self.data["Spar"]
#         Spol = self.data["Spol"]

#         # Find values on either side of C = 1 and interpolate onto 1
#         if len(crel) > 1:
            
#             for i in range(len(crel) - 1):
#                 if np.sign(crel[i] - 1) != np.sign(crel[i + 1] - 1) and i > 0:
#                     interp_par = sp.interpolate.interp1d(
#                         [crel[i], crel[i + 1]], [Spar[i], Spar[i + 1]]
#                     )
#                     interp_pol = sp.interpolate.interp1d(
#                         [crel[i], crel[i + 1]], [Spol[i], Spol[i + 1]]
#                     )

#                     spar_onset = float(interp_par(1))
#                     spol_onset = float(interp_pol(1))
#                     break
#                 if i == len(crel) - 2:
#                     spar_onset = 0
#                     spol_onset = 0
                    
#         if crel.iloc[-1] < 1:
#             # print(Spar)
#             spar_onset_forward = Spar.iloc[-1]
#             spol_onset_forward = Spol.iloc[-1]
                    
#         self.unstable_Lpar_backward = spar_onset
#         self.unstable_Lpol_backward = spol_onset
#         self.unstable_Lpar_forward = spar_onset_forward
#         self.unstable_Lpol_forward = spol_onset_forward
                    
#     def plot_front_movement(self, ax=None, label="", parallel=False, relative=True):
#         if ax is None:
#             fig, ax = plt.subplots()
#         data = self.data

#         if parallel:
#             y = data["Spar"]
#             ylabel = r"$S_{\parallel} [m]$"
#         else:
#             y = data["Spol"]
#             ylabel = "$S_{pol} [m]$"

#         if relative:
#             x = data["cvar"] / data["cvar"].iloc[0]
#             xlabel = "$C_{rel}$"
#         else:
#             x = data["cvar"]
#             xlabel = "C"

#         ax.plot(x, y, marker="o", lw=1, ms=3, label=label)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         ax.set_title("Front movement")

class CoolingCurve:
    def __init__(self, T, Lz, neTau, species, order=8, Tmin=0.5, Tmax=400):
        self.data_original = pd.DataFrame()
        self.data_original["T"] = T
        self.data_original["Lz"] = Lz

        self.neTau = neTau
        self.species = species
        self.order = order
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.resolution = 1000

        self.interpolate()

    def interpolate(self):
        """
        Note that the resulting Tmax may not be the same as specified as input.
        This is because the original data may not have a value exactly at Tmax -
        this means it's clipped to the nearest value under Tmax.
        """

        original_clipped = self.data_original[
            (self.Tmin < self.data_original["T"])
            & (self.data_original["T"] <= self.Tmax)
        ]

        logx = np.log10(original_clipped["T"].values)
        logy = np.log10(original_clipped["Lz"].values)

        coeffs = np.polyfit(logx, logy, self.order)

        def Lz_func(T):
            if T < self.Tmin:
                return 0
            elif T > original_clipped["T"].max():
                return 0
            else:
                return 10 ** np.polyval(coeffs, np.log10(T))

        self.Lz_func = Lz_func
        self.data = pd.DataFrame()
        self.data["T"] = np.linspace(0, self.Tmax + 50, self.resolution)
        self.data["Lz"] = [Lz_func(T) for T in self.data["T"]]

    def plot(self, ax=None, xmax=None, show_fit=False, label="", **kwargs):
        """
        Kwargs are passed to line plot of fitted curve
        """

        if xmax is None:
            xmax = self.Tmax * 1.1

        if ax is None:
            fig, ax = plt.subplots()

        if show_fit:
            ax.plot(
                self.data_original["T"],
                self.data_original["Lz"],
                marker="o",
                lw=0,
                label="Original data",
            )

            label = "Fit"
        elif label != "":
            label = label
        else:
            label = f"{self.species}, $n_{{e}} \\tau = {self.neTau}$"

        ax.plot(self.data["T"], self.data["Lz"], label=label, **kwargs)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("$T_e$ [eV]")
        ax.set_ylabel("Lz [$Wm^3$]")
        ax.legend()
        
def test_scalings(df1, df2, param, cumulative = False):
    """
    Return scaling coefficients to break down impact of different terms on threshold
    df1 is has DLScase stats dataframe for the case before the change and df2 after the change
    """
    diff = df2 / df1
        
    scalings = pd.DataFrame()
    scalings["result"] = diff["cvar"]
    scalings["param_ratio"] = diff[param]
    scalings["simple_noavg_B"] = diff["BxBt"]**(-1) * diff["Lc"]**(-2/7)
    scalings["simple"] = diff["BxBt"]**(-1) * diff["Lc"]**(-2/7) * (diff["avgB_ratio"])**(2/7)
    
    
    if cumulative:
        scalings["abovex"] = scalings["simple"] * diff["W_Tu_simple"]
        scalings["Beff"] = diff["BxBteff"]**(-1) * diff["W_Tu_simple"]
        scalings["W_Tu"] = diff["BxBteff"]**(-1) * diff["W_Tu"]
        scalings["curveclip"] = diff["BxBteff"]**(-1) * diff["W_Tu"] * diff["curveclip"]
        scalings["upstream_rad"] = diff["BxBteff"]**(-1) * diff["W_Tu"] * diff["curveclip"] * diff["upstream_rad"]
    else:  
        scalings["abovex"] = diff["W_Tu_simple"]
        scalings["Beff"] = diff["BxBteff"]**(-1)
        scalings["W_Tu"] = diff["W_Tu"]
        scalings["curveclip"] = diff["curveclip"]
        scalings["upstream_rad"] = diff["upstream_rad"]
        scalings["all"] = diff["BxBteff"]**(-1) * diff["W_Tu"] * diff["curveclip"] * diff["upstream_rad"]

    return scalings


class plot_comparison():
    def __init__(self, comps, studies):
        self.comps = comps
        self.studies = studies
        # self.studies = list(comps.keys())
        self.cases = comps[self.studies[0]].keys()
        
        self.study_labels = dict(
            adas_curve = r"Constant $n_e \tau$ cooling curve",
            fit_curve = "Extract cooling curve",
            fit_curve_kappa = r"Extract cooling curve, tune $\kappa_{0,e}$",
            other_losses = "Total rad. losses",
            include_cz = "Fix $f_{N}$",
            include_cz_ne = "Fix $f_{N}n_{e}^{2}$"
        )
        
        # self.DLSstyles = dict(
        # adas_curve =    dict(lw = 0, marker = "o", ms = 2, markerfacecolor = "None"),
        # fit_curve =     dict(lw = 0, marker = "x", ms = 4, markeredgewidth = 1, markerfacecolor = "None"),   
        # other_losses =     dict(lw = 0, marker = "p", ms = 4, markeredgewidth = 1, markerfacecolor = "None"),   
        # include_cz =    dict(lw = 0, marker = "s", ms = 3, markeredgewidth = 1, markerfacecolor = "None"),   
        # include_cz_ne = dict(lw = 0, marker = "1", ms = 6, markeredgewidth = 1, markerfacecolor = "None"),   
        # )
        
        self.DLSstyles = dict(
        adas_curve =       dict(lw = 1.5, ls = "--", marker = "o", ms = 0, markerfacecolor = "None"),
        fit_curve =        dict(lw = 1.5, ls = "--", marker = "x", ms = 0, markeredgewidth = 1, markerfacecolor = "None"),   
        fit_curve_kappa =  dict(lw = 1.5, ls = "--", marker = "p", ms = 0, markeredgewidth = 1, markerfacecolor = "None"),   
        include_cz =    dict(lw = 0, marker = "s", ms = 3, markeredgewidth = 1, markerfacecolor = "None"),   
        include_cz_ne = dict(lw = 0, marker = "1", ms = 6, markeredgewidth = 1, markerfacecolor = "None"),   
        )

        self.SOLPSstyle = dict(lw = 1.0, ms = 0, c = "darkslategrey")
        self.Xpointstyle = dict(ls = "-", lw = 1, alpha = 0.5)
        self.figure_labels = ["a)", "b)", "c)"]
        self.ylabels = {
            "Te" : r"eV",
            "Ne" : r"$m^{-3}$",
            "Ne_sq_cz" : r"$m^{-6}$",
            "qpar" : r"$MWm^{-2}$",
            "Qrad_cum_norm" : r"fraction",
            "qpar_cond_frac" : r"fraction",
        }
        self.titles = {
            "Qrad_cum_norm" : "Cum. rad. integral (normalised)",
            "Te" : "Electron temp.",
            "Ne" : "Electron dens. (normalised)",
            "Ne_sq_cz" : f"$N_e^2 f_Ar$ (normalised)",
            "qpar" : "Tot. par. heat flux",
        }         

        # self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # self.colors = ["darkorange", "firebrick"]
        self.colors = ["teal", "darkorange", "firebrick"]
        self.dls_skip = 4
    
        
        
    def make_grid(self, ax):
        ax.grid(which = "major", visible = True, c = "k", alpha = 0.15)
        ax.grid(which = "minor", visible = True, c = "k", alpha = 0.075)
        
        
    def make_legend_elements(self):
        elements = [mpl.lines.Line2D([0],[0], **self.SOLPSstyle, label = "SOLPS")]
        for i, study in enumerate(self.studies):
            elements.append(mpl.lines.Line2D([0],[0], **self.DLSstyles[study], color = self.colors[i], label = self.study_labels[study]))
        elements.append(mpl.lines.Line2D([0],[0], **self.Xpointstyle, color = "deeppink", label = "X-point"))
        
        return elements
        
    def plot(self, param, axes, normalise = False):
        
        for i, case in enumerate(self.cases):
            ax = axes[i]
            
            for study_no, study in enumerate(self.studies):
                comp = self.comps[study][case]
                dls = comp.dls.copy()
                solps = comp.solps.copy()  
                
                if param == "qpar_cond_frac":
                    dls["qpar_cond_frac"] = 1
                
                if normalise:
                    dls_data = dls[param] / dls.iloc[-1][param]
                    solps_data = solps[param] / solps.iloc[-1][param]
                else:
                    dls_data = dls[param]
                    solps_data = solps[param]
                    
                # Spar_interp = np.linspace(0, dls.iloc[-1]["Spar"], 50)
                # dls_data_interp = [sp.interpolate.interp1d(dls["Spar"], dls_data)(s) for s in Spar_interp]
                
                if param == "qpar": 
                    mult = 1e-6
                    if solps_data.max() < 0:  # Correct for -ve hflux 
                        solps_data *= -1
                else:
                    mult = 1
                
                ax.plot(solps["Spar"], solps_data*mult, label = study, **self.SOLPSstyle)  # SOLPS
                ax.plot(dls["Spar"], dls_data*mult, **self.DLSstyles[study], c = self.colors[study_no], alpha = 0.8)   # DLS
                
                # if param == "qpar":
                #     ax.plot(solps["Spar"], solps["qpar_cond"]*mult, label = study, **{**dict(ls=":"), **self.SOLPSstyle})  # SOLPS
                
            ## Make X-point
            ax.set_xlabel(r"$S_{\parallel}$ [m]")
            ax.set_ylabel(self.ylabels[param])
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
            ax.grid(which = "both", visible = True, lw = 0.16, c = "k", alpha = 0.3)
            # ax.make_xpoint()
            
            if any([x in param for x in ["Ne"]]):
                logscale = True
                ax.set_yscale("log")
            else:
                logscale = False
            
            Spar_x = solps[solps["Xpoint"]==1]["Spar"].values[0]
            ylim = ax.get_ylim()
            ax.vlines(Spar_x, -100, 1e30, **self.Xpointstyle, colors = "deeppink", label = "X-point")
            
                
            # ax.text(0.55, 0.5, "X-point", transform = ax.transAxes,
            #         fontsize = "xx-small", verticalalignment="top", 
            #         color = "deeppink", rotation = "vertical", alpha = 0.8)

            ax.set_ylim(ylim)
                
                
            
        
    # def make_title(self, ax, case):
        # ax.set_title(figure_labels[i], loc = "right", y = 0.02, x = 0.97, fontsize = 20)
        
        
class scaling_comparison():
    def __init__(self, out1, out2, param):
        
        
        # Compare last and first case only!!

        df1 = pd.DataFrame([FrontLocation(out1).stats])
        df2 = pd.DataFrame([FrontLocation(out2).stats])
        param_ratio = (df2[param] / df1[param]).squeeze()
        terms = pd.concat([df1, df2], axis = 0)
        terms.index = ["base", "test"]
        terms = terms.T
        
        terms_check = pd.DataFrame()
        terms_check.loc["total", "cvar_ratio"] = (terms.loc['cvar', 'test'] / terms.loc['cvar', 'base'])**(-1)
        
        
        terms["ratio"] = (terms["test"] / terms["base"])**(-1)   # Remember that K is positive for lower threshold
        terms_full = terms.copy()
        terms = terms.loc[["Beff", "W_Tu", "curveclip", "upstream_rad"],:]
        terms["log10(ratio)"] = np.log10(terms["ratio"])
        terms["abs(log10(ratio))"] = terms["log10(ratio)"].abs()
        terms["fraction"] = terms["log10(ratio)"] / terms["abs(log10(ratio))"].sum()
        
        ## Plot needs the positive bits to sum up to K. 
        # Take fraction (sums to 1) and scale to K. now sum(abs(weights)) = K.
        # Then divide by the sum of the positive fractions. Now sum of positive weights = K, and 
        # the negative weight is scaled the same as the positive weights for consistency.
        K = np.log(terms["ratio"].product()) / np.log(param_ratio)
        terms["weights"] = terms["fraction"] * K / terms[terms>0]["fraction"].sum()   

        
        terms_check.loc["total", "scaling_ratio"] = terms["ratio"].product(axis=0)
        terms_check.loc["total", "abs(fraction)"] = terms["fraction"].abs().sum(axis=0)
        terms_check.loc["total", "weights"] = terms["weights"].sum(axis=0)
        terms_check.loc["total", "abs(weights)"] = terms["weights"].abs().sum(axis=0)
        terms_check.loc["total", "weights_pos"] = terms[terms > 0]["weights"].sum(axis=0)
        terms_check.loc["total", "K"] = K
        
        self.df1 = df1
        self.df2 = df2
        self.terms = terms
        self.terms_full = terms_full
        self.terms_check = terms_check
        
        
def compare_dls(toplot, params, settings = defaultdict(dict), xlims = defaultdict(dict), scales = defaultdict(dict)):

    no_figs = len(params)
    fig, axes = plt.subplots(1,no_figs,figsize=(3*no_figs,3))

    for i, name in enumerate(toplot):
        df = toplot[name]

        for j, param in enumerate(params):
            axes[j].plot(df["Spar"], df[param], label = name, **settings[param])
            axes[j].set_title(param)
            if param in xlims:
                axes[j].set_xlim(xlims[param])
            
            if param in scales:
                axes[j].set_yscale(scales[param])

        for ax in axes:
            ax.legend()