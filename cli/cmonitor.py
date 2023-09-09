# Reading with cache for extra speed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import boutdata
from boututils.options import BOUTOptions
from boutdata.collect import create_cache
import argparse
import os




def cmonitor(path, save = False, noshow = False):
    
    if path == ".":
        casename = os.path.basename(os.getcwd())
    else:
        casename = os.path.basename(path)
    print(casename)
    
    cache = create_cache(path, "BOUT.dmp")
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

    # Get process parameters
    t = get_var("t") * (1/Omega_ci) * 1000
    Ne = get_var("Ne") * Nnorm
    Nn = get_var("Td") * Nnorm
    Te = get_var("Te") * Tnorm
    Tn = get_var("Td") * Tnorm

    # Get solver parameters
    wtime = get_var("wtime")
    nliters = get_var("cvode_nliters")
    nniters = get_var("cvode_nniters")
    nfails = get_var("cvode_num_fails")
    lorder = get_var("cvode_last_order")

    # Calculate locations
    # [t, x, y]
    j2_2g = jyseps2_2 + MYG * 3
    j1_2g = jyseps1_2 + MYG * 3
    y_omp = int((j2_2g - j1_2g) / 2)
    x_sep = ixseps1
    x_ng = slice(2,-2)   # No guards in  X

    # First row of plots
    Ne_omp = Ne[:,x_sep,y_omp]
    Ne_target = np.max((0.5*(Ne[:,x_ng, -2] + Ne[:,x_ng,-3])), axis = 1)
    Nn_target = np.max((0.5*(Nn[:,x_ng, -2] + Nn[:,x_ng,-3])), axis = 1)
    Tn_sol = Tn[:, -3, y_omp]

    # Second row of plots
    stime = np.diff(t)
    wtime_per_stime = wtime[1:]/stime
    lratio = np.diff(nliters) / np.diff(nniters)   # Ratio of linear to nolinear iterations

    # Plotting
    scale = 1.2
    figsize = (8*scale,4*scale)
    dpi = 150/scale
    fig, axes = plt.subplots(2,4, figsize=figsize, dpi = dpi)
    
    fig.subplots_adjust(hspace=0.4, top = 0.85)
    fig.suptitle(casename, y = 0.95)

    lw = 2
    axes[0,0].plot(t, Ne_omp, c = "darkorange", lw = lw)
    axes[0,0].set_title("$N_{e}^{omp}$")
    axes[0,1].plot(t, Ne_target, c = "darkorchid", lw = lw)
    axes[0,1].set_title("$N_{e}^{targ}$")
    axes[0,2].plot(t, Nn_target, c = "deeppink", lw = lw)
    axes[0,2].set_title("$N_{n}^{targ}$")
    axes[0,3].plot(t, Tn_sol, c = "limegreen", lw = lw)
    axes[0,3].set_title("$T_{n}^{omp,sol}$")
        
    axes[1,0].plot(t[1:], wtime_per_stime, c = "k", lw = lw)
    axes[1,0].set_title("wtime/stime")
    axes[1,1].plot(t[1:], lratio, lw = lw, c = "k")
    axes[1,1].set_title("linear/nonlinear")
    axes[1,2].plot(t[1:], np.diff(nfails), c = "k", lw = lw)
    axes[1,2].set_ylim(0,None)  # Messes up when you restart and fail counter resets
    axes[1,2].set_title("nfails")
    axes[1,3].plot(t, lorder, c = "k", lw = lw)
    axes[1,3].set_title("order")

    for i in [0,1]:
        for ax in axes[i,:]:
            ax.grid(c = "k", alpha = 0.15)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.tick_params(axis='x',labelsize=8)
            
    fig.tight_layout()
    
    if noshow is False:
        plt.show()  
    
    if save:
        fig.savefig(f"mon_{casename}.png")
        

        
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
        
# Define arguments
parser = argparse.ArgumentParser(description = "Case monitor")
parser.add_argument("path", type=str, help = "Path to case")
parser.add_argument("-save", action="store_true")
parser.add_argument("-noshow", action="store_true")
# Extract arguments and call function
args = parser.parse_args()

cmonitor(args.path, save = args.save, noshow = args.noshow)