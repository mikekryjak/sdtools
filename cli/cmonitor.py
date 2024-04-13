#!/usr/bin/env python3

# Reading with cache for extra speed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import boutdata
from boututils.options import BOUTOptions
from boutdata.collect import create_cache
import argparse
import os
import time as tm




def cmonitor(path, save = False, plot = False, table = True, neutrals = False):
    """
    Produce convergence report of 2D Hermes-3 simulation
    Plots of process conditions at OMP and target 
    As well as solver performance indices
    In addition to the plot, a CLI friendly table can be produced
    
    Inputs
    -----
    path: path to case directory
    save: bool, save figure. saved by case name
    plot: bool, show ploy
    table = bool, show table
    """
    tstart = tm.time()
    
    if path == ".":
        casename = os.path.basename(os.getcwd())
    else:
        casename = os.path.basename(path)
    print(f"Reading {casename}")
    print("Calculating...", end = "")
    
    # Reading with cache for extra speed
    cache = create_cache(path, "BOUT.dmp")
    print("..cache", end="")
    def get_var(name):
        return  boutdata.collect(
            name,
            path = path,
            yguards = True,  # Akways with guards to minimise mistakes
            xguards = True,  # Always with guards to minimise mistakes
            strict = True,   # To prevent reading wrong variable by accident
            info = False,
            datafile_cache = cache
        ).squeeze()

    # Get normalisations and geometry
    Nnorm = get_var("Nnorm")
    Tnorm = get_var("Tnorm")
    Omega_ci = get_var("Omega_ci")
    MYG = get_var("MYG")
    ixseps1 = get_var("ixseps1")
    jyseps1_2 = get_var("jyseps1_2")
    jyseps2_2 = get_var("jyseps2_2")
    dx = get_var("dx")
    dy = get_var("dy")
    dz = get_var("dz")
    J = get_var("J")
    
    
    
    # Get process parameters
    t = get_var("t") * (1/Omega_ci) * 1000
    Ne = get_var("Ne") * Nnorm
    Nn = get_var("Nd") * Nnorm
    Te = get_var("Te") * Tnorm
    Tn = get_var("Td") * Tnorm
    
    res = {}
    
    for param in ["ddtPe","ddtPi", "ddtPn", "ddtNe", "ddtNn", "ddtNVi", "ddtNVd"]:
        try:
            res[param] = get_var(param)
        except:
            res[param] = np.zeros_like(Ne)
    

    dv = dx * dy * dz * J

    

    # Get solver parameters
    wtime = get_var("wtime")
    nliters = get_var("cvode_nliters")
    nniters = get_var("cvode_nniters")
    nfails = get_var("cvode_num_fails")
    lorder = get_var("cvode_last_order")
    
    print("..data", end="")

    # Calculate locations
    # [t, x, y]
    j2_2g = jyseps2_2 + MYG * 3
    j1_2g = jyseps1_2 + MYG * 3
    y_omp = int((j2_2g - j1_2g) / 2) + j1_2g
    x_sep = ixseps1
    x_ng = slice(2,-2)   # No guards in  X

    # First row of plots
    Ne_sep = Ne[:,x_sep,y_omp]
    # Ntot = ((Ne[:, x_ng, :] + Nn[:, x_ng, :]) * dv).sum(axis = (1,2))
    Ne_target = np.max((0.5*(Ne[:,x_ng, -2] + Ne[:,x_ng,-3])), axis = 1)
    Nn_target = np.max((0.5*(Nn[:,x_ng, -2] + Nn[:,x_ng,-3])), axis = 1)
    Tn_target = np.max((0.5*(Tn[:,x_ng, -2] + Tn[:,x_ng,-3])), axis = 1)
    Te_target = np.max((0.5*(Te[:,x_ng, -2] + Te[:,x_ng,-3])), axis = 1)
    Tn_sol = Tn[:, -3, y_omp]
    Te_sol = Te[:, -3, y_omp]

    def append_first(x):
        return np.insert(x,0,x[0])
    # Second row of plots
    stime = np.diff(t, prepend = t[0]*0.99)
    ms_per_24hrs = (stime) / (wtime/(60*60*24))  # ms simulated per 24 hours
    lratio = np.diff(nliters, prepend=nliters[1]*0.99) / np.diff(nniters, prepend=nniters[1]*0.99)   # Ratio of linear to nolinear iterations
    fails = np.diff(nfails, prepend = nfails[1]*0.99)
    fails[0] = fails[1]
    lorder[0] = lorder[1]
    ms_per_24hrs[0] = ms_per_24hrs[1]
    
    # ddt
    res = {}
    for param in res:
        res[param] = (res[param] * dv) / np.sum(dv)   # Volume weighted
        res[param] = np.sqrt(np.mean(res[param]**2, axis = (1,2)))  # RMS
        res[param] = np.convolve(res[param], np.ones(1), "same")    # Moving average with window of 1
    
    print("..calculations", end="")
    
    # Plotting
    if plot is True or save is True:
        scale = 1.2
        figsize = (8*scale,4*scale)
        dpi = 150/scale
        fig, axes = plt.subplots(2,4, figsize=figsize, dpi = dpi)

        fig.subplots_adjust(hspace=0.4, top = 0.85)
        fig.suptitle(casename, y = 1.02)

        lw = 2
        axes[0,0].plot(t, Ne_sep, c = "darkorange", lw = lw)
        axes[0,0].set_title("$N_{e}^{omp,sep}$")
        
        if neutrals is True:
            
            axes[0,3].plot(t, Tn_sol, c = "limegreen", lw = lw)
            axes[0,3].set_title("$T_{n}^{omp,sol}$")
            
            axes[0,1].plot(t, Tn_target, c = "darkorchid", lw = lw)
            axes[0,1].set_title("$T_{n}^{targ,max}$")

            axes[0,2].plot(t, Nn_target, c = "deeppink", lw = lw)
            axes[0,2].set_title("$N_{n}^{targ,max}$")
        
        else:
            
            axes[0,1].plot(t, Te_sol, c = "limegreen", lw = lw)
            axes[0,1].set_title("$T_{e}^{omp,sol}$")
            
            axes[0,2].plot(t, Ne_target, c = "deeppink", lw = lw)
            axes[0,2].set_title("$N_{e}^{targ,max}$")
            
            axes[0,3].plot(t, Te_target, c = "darkorchid", lw = lw)
            axes[0,3].set_title("$T_{e}^{targ,max}$")
            
            
            
            
        axes[1,0].plot(t, ms_per_24hrs, c = "k", lw = lw)
        axes[1,0].set_title("ms $t_{sim}$ / 24hr $t_{wall}$")
        # axes[1,0].set_yscale("log")
        axes[1,1].plot(t, lratio, c = "k", lw = lw)
        axes[1,1].set_title("linear/nonlinear")
        axes[1,2].plot(t, np.clip(fails, 0, np.max(fails)), c = "k", lw = lw)
        axes[1,2].set_title("nfails")
        axes[1,2].set_ylim(0,None)
        # axes[1,2].set_yscale("log")
        axes[1,3].plot(t, lorder, c = "k", lw = lw)
        axes[1,3].set_title("order")

        for i in [0,1]:
            for ax in axes[i,:]:
                ax.grid(c = "k", alpha = 0.15)
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=3, nbins=5))
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=3, nbins=5))
                # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(4))
                ax.tick_params(axis='x',labelsize=8)
                ax.tick_params(axis='y',labelsize=8)
                
        fig.tight_layout()
        
        print("..figures", end="")
        
        if plot is True:
            plt.show()  
            
        if save:
            fig.savefig(f"mon_{casename}.png", bbox_inches="tight", pad_inches = 0.2)
            print("..saved figures", end="")

    ### Print table

    if table is True:
        
        def pad(x, l):
            num_spaces = l - len(x)
            return  str(x)+" "*num_spaces

        def pad_minus(x):
            x = f"{x:.3e}"
            pad = "" if x[0] == "-" else " "
            return pad+x


        # Nseprate = np.diff(Ne_sep, prepend=Ne_sep[0]) / Ne_sep
        Nseprate = np.gradient(Ne_sep, t) / Ne_sep

        print("..table\n\n")
        print(f"it |  t[ms] |    Nsep      Nseprate |   Netarg  |   Nntarg  | Tnsol | w/s time | l/n |  nfails  | order |")
        # print("~"*100)
        for i, time in enumerate(t):
            Tnprint = pad(f"{Tn_sol[i]:.1f}",5)
            s1=f"{pad(str(i),2)} | {time:.2f} | {Ne_sep[i]:.3e}  {pad_minus(Nseprate[i])} | {Ne_target[i]:.3e}"
            s2=f" | {Nn_target[i]:.3e} | {Tnprint} | {wtime_per_stime[i]:.2e} | {lratio[i]:.1f} | {nfails[i]:.2e} |   {lorder[i]:.0f}   |"
            
            print(s1+s2)
        
    tend = tm.time()
    print(f"Executed in {tend-tstart:.1f} seconds")
         
            
        

        
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------

if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Case monitor")
    parser.add_argument("path", type=str, help = "Path to case")
    parser.add_argument("-p", action="store_true", help = "Plot?")
    parser.add_argument("-t", action="store_true", help = "Table?")
    parser.add_argument("-s", action="store_true", help = "Save figure?")
    parser.add_argument("-neutrals", action="store_true", help = "Neutral focused plot?")
    
    # Extract arguments and call function
    args = parser.parse_args()
    cmonitor(args.path, plot = args.p, table = args.t, save = args.s, neutrals = args.neutrals)