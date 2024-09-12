
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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

        df.loc[i, "cvar"] = s["cvar_trim"][i]
        df.loc[i, "crel"] = s["crel_trim"][i]
        
        df.loc[i, "Btot"] = np.interp(df.loc[i, "Spar"], S, Btot)
        
        df.loc[i, "Btot_eff"] = Beff
        df.loc[i, "Btot_eff_old"] = Beff_old
    
        
        df.loc[i, "Tu"] = s["Tprofiles"][i][-1]
    
    return df

def get_detachment_scalings(profiles, stores, kappa0 = 2500):
    

    ## NOTE: Using power in W/m2 because we are using this calc to weigh Beff!
    front_dfs = [get_front_widths(profiles[x], stores[x], Rcutoff = 0.5, use_prad = True) for x in profiles]

    df = pd.DataFrame()
    df["thresholds"] = np.array([stores[x]["threshold"] for x in profiles])
    df["windows"] = np.array([stores[x]["window_ratio"] for x in profiles]) 
    df["L"] = np.array([profiles[x].get_connection_length() for x in profiles])
    df["BxBt"] = np.array([profiles[x].get_total_flux_expansion() for x in profiles])
    df["Bf"] = np.array([profiles[x]["Btot"][0] for x in profiles])
    df["frac_gradB"] = np.array([profiles[x].get_average_frac_gradB() for x in profiles])
    df["avgB_ratio"] = np.array([profiles[x].get_average_B_ratio() for x in profiles])
    df["BxBt_eff"] = [df["Btot_eff"].iloc[-1] / df["Btot_eff"].iloc[0] for df in front_dfs]
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

        
        
        
        ### Old derivation terms
    
        ## C0

        ## Cooling curve integral
        # Effect of Tu clipping the integral in wide curves
        df.loc[i, "C0"] = 7**(-2/7) * (2*kappa0)**(-3/14) * (sp.integrate.trapezoid(y = Lz*T**0.5, x = T))**(-0.5)
        # df.loc[i, "C0"] = 7**(-2/7) * (2*kappa0)**(-3/14) * (sp.integrate.trapezoid(y = Lz*T**0.5 * Btot**(-2), x = T))**(-0.5)
    
        ## Classic DLS thresholds
        df.loc[i, "DLS_thresholds"] = CfInt(
                                            spar = profile["S"], 
                                            B_field = profile["Btot"], 
                                            sx = profile["S"][profile["Xpoint"]], 
                                            L = profile["S"][-1],
                                            sh = store["Splot"][0]
                                        )
        
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
    axes[1,2].set_ylabel("$S_{front, final} \ /\  S_{pol, x}$")
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
        if idx == 0: BxBtkwargs["label"] = "$B_{X} \/ B_{t}$"
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
        
    
            
        
# def plot_designs(profiles, colors = None, basis = "Spol", mode = "outer"):
#     fig, axes = plt.subplots(2,3, figsize = self.figsize)
#     titlesize = "x-large"
#     lw = 3
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors == None else colors
#     spines = True
    
    
#     for i, key in enumerate(profiles.keys()):
#         p = profiles[key]
#         ax = axes[0,0]
#         ax.plot(p.R - p.R[p.Xpoint], p.Z - p.Z[p.Xpoint], label = key, color = colors[i], lw = lw)
#         ax.set_aspect("equal")
#         ax.set_ylim(-3.0, 1)
#         ax.set_title("Design", fontsize = titlesize)
#         ax.set_xlabel(r"$R - R_{Xpoint}\ [m]$")
#         ax.set_ylabel(r"$Z - Z_{Xpoint}\ [m]$")
#         ax.legend()
        
#         selector = slice(None,p.Xpoint)
        
#         xplot = (p[basis] - p[basis][p.Xpoint])[selector] 
        
#         ## Reverse X so that inner can go from left to right
#         # Distance still m from X-point
#         # if mode == "inner":
#         #     xplot *= -1
            
#         xplot *= -1
            
#         xlabel = dict(
#             Spol = r"$S_{\theta}\ [m]$",
#             S = r"$S_{\parallel}\ [m]$"
#         )[basis]
            
        
        
#         ax = axes[1,0]
#         L = [p.get_connection_length() for p in profiles.values()]
#         L = L/L[0]
#         BxBt = [p.get_total_flux_expansion() for p in profiles.values()]
#         BxBt /= BxBt[0]
#         Lkwargs = dict(zorder = 100, alpha = 1, color = colors[i], lw = 0, marker = "o", ms = 10,  ls = "-")
#         BxBtkwargs = dict(zorder = 100, alpha = 1, color = colors[i], lw = 0, marker = "x", ms = 10, markeredgewidth = 5, ls = "-")
        
#         if i == 0: Lkwargs["label"] = "Lc"
#         if i == 0: BxBtkwargs["label"] = "$B_{X} \/ B_{t}$"
#         ax.plot(i, L[i], **Lkwargs)
#         ax.plot(i, BxBt[i], **BxBtkwargs)
#         ax.set_title("Flux expansion and connection length", fontsize = titlesize)
#         ax.legend(bbox_to_anchor = (0.3,0.95))
        
#         ax = axes[0,1]
#         ax.plot(xplot, p.Btot[selector], label = key, color = colors[i], lw = lw)
#         ax.set_title(r"Total field", fontsize = titlesize)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(r"$B_{tot}\ [T]$")
        
#         ax = axes[1,1]
#         ax.plot(xplot, p.Bpol[selector], label = key, color = colors[i], lw = lw)
#         ax.set_title(r"Poloidal field", fontsize = titlesize)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(r"$B_{\theta}\ [T]$")
        
#         ax = axes[0,2]
#         ax.plot(xplot, (np.gradient(p.Btot, p.S)/p.Btot)[selector], label = key, color = colors[i], lw = lw)
#         ax.set_title(r"Parallel B gradient", fontsize = titlesize)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(r"$\frac{1}{B} \frac{dB}{dS_{\parallel}}\ [T]$")
        
#         ax = axes[1,2]
#         ax.plot(xplot, (p.Bpol/p.Btot)[selector], label = key, color = colors[i], lw = lw)
#         ax.set_title(r"Field pitch", fontsize = titlesize)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(r"$B_{\theta} / B_{tot}$")
        
#     # Reverse axes so inner goes left to right
#     if mode == "inner":
#         for ax in [axes[0,1], axes[1,1], axes[0,2], axes[1,2]]:
#             ax.set_xlim(ax.get_xlim()[::-1])
        
#     fig.tight_layout()
    
#     if spines is True:
#         for spine in ["top", "left", "right", "bottom"]:
#             ax.spines[spine].set_visible(True)


def compare_SOLPS_DLS(slc, out):

    df = slc.get_1d_poloidal_data(["hx", "hz", "R", "Te", "RAr", "ne", "fhe_cond", "fhe_32", "fhi_cond", "fhi_32", "fhe_thermj"], sepadd = 1)
    
    for param in df:
        if param.startswith("fh"):
            df[param] /= (df["hx"] * (2*np.pi*df["R"]))
    
    for param in ["fhe_32", "fhi_32"]:
        df[param] *= 5/3
    
        
    df["fhe_total"] = df["fhe_cond"] + df["fhe_32"] + df["fhe_thermj"]
    df["fhi_total"] = df["fhi_cond"] + df["fhi_32"]
    df["fh_total"] = df["fhe_total"] + df["fhi_total"]

    dfx = df[df["Xpoint"] == 1]

    front_satisfied = (abs(df["RAr"]) < 0.001*abs(df["RAr"]).max()).abs()
    front_satisfied = front_satisfied[::-1]
    front_idx = 0
    for i in front_satisfied.index[::-1]:
        if front_satisfied[i] == False:
            front_idx = i-1
            
    dffront = df.loc[front_idx]

    ## DLS
    dlsR = out["Rprofiles"][0]
    dlsSpar = out["Sprofiles"][0][::-1]
    # dlsSpar = dlsSpar[-1] - dlsSpar 
    dlsTe = out["Tprofiles"][0]
    dlsqpar = out["Qprofiles"][0]
    dlsNe = out["cvar"][0] * dlsTe[-1] / dlsTe   ## Assuming cvar is ne

    nfigs = 5
    fsize = 4
    fig, axes =plt.subplots(1,nfigs, figsize = (nfigs*fsize, fsize))

    ax = axes[0]
    ax.plot(dlsSpar, dlsR, label = "DLS")
    ax.plot(df["Spar"], df["RAr"], label = "SOLPS")


    ax.set_title("Total Argon $Q_{rad}$ [W/m3]")
    ax.set_yscale("log")

    ax = axes[1]
    ax.set_title("Cumulative Argon $Q_{rad}$ [W/m3]")

    cumR = dlsR.cumsum() / dlsR.cumsum().max()
    ax.plot(dlsSpar, cumR, label = "DLS")

    cumR = abs(df["RAr"][::-1]).cumsum() / abs(df["RAr"][::-1]).cumsum().max()
    ax.plot(df["Spar"][::-1], cumR, label = "SOLPS")
    
    ax = axes[2]
    ax.set_title("$q_{\parallel}$")
    ax.plot(dlsSpar, dlsqpar, label = "DLS")
    ax.plot(df["Spar"], df["fhe_cond"], label = "SOLPS electron cond")
    ax.plot(df["Spar"], df["fh_total"], label = "SOLPS total")
    ax.set_yscale("log")

    ax = axes[3]
    ax.set_title("Te")
    ax.plot(dlsSpar, dlsTe, label = "DLS")
    ax.plot(df["Spar"], df["Te"], label = "SOLPS")

    ax = axes[4]
    ax.set_title("Ne")
    ax.plot(dlsSpar, dlsNe, label = "DLS")
    ax.plot(df["Spar"], df["ne"], label = "SOLPS")
    ax.set_yscale("log")


    for ax in axes:
        ax.set_xlabel("Lpar [m]")
        ylims = ax.get_ylim()
        ax.vlines(dfx["Spar"], *ylims, color = "darkslategrey", ls = "-", alpha = 0.5, lw = 1, label = "X-point")
        ax.vlines(dffront["Spar"], *ylims, color = "deeppink", ls = "-", alpha = 0.3, lw = 1, label = "Front")
        ax.set_ylim(ylims)
        ax.legend(fontsize = "x-small")

    # fig.tight_layout()
    
