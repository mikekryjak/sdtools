
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from ThermalFrontFormulation import *

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
          
    fig, axes = plt.subplots(1,4,figsize = (15*len(list_idx)/4,4))
    
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
        
        # ax1.axvline(x = x[Xpoint], c = "k", ls = ":", lw = 1.5)

        ## Axis 2, radiation integral
        ax2 = axstore[i]["ax2"] = ax1.twinx()
        c = "blue"
        ax2.plot(x, Rchoice, color = c, ls = "-", label = "Cumulative radiation integral", lw = 2,   alpha = 1)
        ax2.set_ylabel("Cumulative radiation fraction")
        ax2.spines["right"].set_color(c)
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
    df["frac_gradB"] = np.array([profiles[x].get_average_frac_gradB() for x in profiles])
    df["avgB_ratio"] = np.array([profiles[x].get_average_B_ratio() for x in profiles])
    df["BxBt_eff"] = [df["Btot_eff"].iloc[-1] / df["Btot_eff"].iloc[0] for df in front_dfs]
    df["BxBt_eff_old"] = [df["Btot_eff_old"].iloc[-1] / df["Btot_eff_old"].iloc[0] for df in front_dfs]
    df["Lx"] = [profiles[x]["S"][profiles[x]["Xpoint"]] for x in profiles]
    df["Tu"] = [stores[x]["Tprofiles"][0][-1] for x in profiles]

    ## Get effective avgB ratio
    avgB_ratio_eff = []
    for fdf in front_dfs:
        newS = np.linspace(fdf["Spar"].iloc[0], fdf["Spar"].iloc[-1], 100)
        Btot_eff_interp = sp.interpolate.make_interp_spline(fdf["Spar"], fdf["Btot_eff"])(newS)
        avgB_ratio_eff.append(fdf["Btot"].iloc[-1] / np.mean(Btot_eff_interp))
        
    df["avgB_ratio_eff"] = avgB_ratio_eff
    
    ### Get C0, classic DLS thresholds
    for i, key in enumerate(stores):
        store = stores[key]
        profile = profiles[key]
        S = profile["S"]   # Careful, this is the input profile not the profile the DLS ran with which is refined
        Lpar = S[-1]
        Xpoint = profile["Xpoint"]
        Btot = store["Btotprofiles"][0]
        Sx = S[Xpoint]
        Bx = Btot[Xpoint]
        qpar = store["Qprofiles"][0]
        Lfunc = store["constants"]["Lfunc"]
    
        ## C0
        x = store["Sprofiles"][0]
        T = pad_profile(x, store["Tprofiles"][0])
        Lz = [Lfunc(x) for x in T]


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