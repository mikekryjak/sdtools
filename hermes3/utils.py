import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pandas as pd


class HiddenPrints:
    """
    Suppress printing by:
    with HiddenPrints():
        ...
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def create_norm(logscale, norm, vmin, vmax, linthresh = None):
    if logscale:
        if norm is not None:
            raise ValueError(
                "norm and logscale cannot both be passed at the same time."
            )
        if vmin * vmax > 0:
            # vmin and vmax have the same sign, so can use standard log-scale
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            # vmin and vmax have opposite signs, so use symmetrical logarithmic scale
            if not isinstance(logscale, bool):
                linear_scale = logscale
            else:
                linear_scale = 1.0e-5
            
            if linthresh is None:
                linear_threshold = min(abs(vmin), abs(vmax)) * linear_scale
                if linear_threshold == 0:
                    linear_threshold = 1e-4 * vmax   # prevents crash on "Linthresh must be positive"
            else:
                linear_threshold = linthresh
            norm = mpl.colors.SymLogNorm(linear_threshold, vmin=vmin, vmax=vmax)
    elif norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return norm


def read_file(filename, quiet = False):
# Reads a pickle file and returns it.

    with open(filename, "rb") as f:
    # Open file in read binary mode, dump file to result.
        data = pkl.load(f)
        if not quiet:
            print("{} loaded".format(filename))
        
    return data

def write_file(data, filename, quiet = False):
# Writes an object to a pickle file.
    
    with open(filename, "wb") as file:
    # Open file in write binary mode, dump result to file
        pkl.dump(data, file)
        if not quiet:
            print("{} written".format(filename))

def constants(name):
    
    d = dict()
    d["mass_p"] = 1.6726219e-27 # Proton mass [kg]
    d["mass_e"] = 9.1093837e-31 # Electron mass [kg]
    d["a0"] = 5.29177e-11 # Bohr radius [m]
    d["q_e"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    d["k_b"] = 1.3806488e-23 # Boltzmann self.ant [JK^-1]
    d["e0"] = 8.854187817e-12 # Vacuum permittivity [Fm^-1]
    
    return d[name]


def mike_cmap():
    return ["teal", "darkorange", "firebrick", "limegreen", "mediumblue", "darkorchid", "deeppink"]


def select_custom_core_ring(ds, i):
    sel = {"x":i, "theta":slice(ds.regions["core"].ylower_ind, ds.regions["core"].yupper_ind)}
    return ds[sel]
   
def select_custom_sol_ring(ds, i):
    sel = {"x":i, "theta":slice(ds.regions["lower_inner_pfr"].ylower_ind, ds.regions["lower_outer_pfr"].yupper_ind)}
    return ds[sel] 
    
def make_cmap(cmap, N):
    """
    Extract discrete colors from a continuous colormap
    
    Parameters
    ----------
    N = number of colors
    cmap = Matplotlib colormap name
    """
    return plt.cm.get_cmap(cmap)(np.linspace(0, 1, N))

def display_dataframe(df, format = "{:.2e}"):

    def styler(s):
            if abs(s) < 0.01 or pd.isna(s):
                c =  "color: lightgrey"
            else:
                c =  "color: black"

            return c
            
    ts = df.style.format("{:.2e}")
    ts = ts.applymap(styler)
    display(ts)
    
def guard_replace_1d(da):
    """
    Replace the inner guard cells with the values of their respective
    cell edges, i.e. the values at the model inlet and at the target.
    This is done by interpolating the value between the two neighbouring
    cell centres.

    Cell order at target:
    ... | last | guard | second guard (unused)
                ^target      
        |  -3  |  -2   |      -1
        
    Returns
    ----------
    - Numpy array with guard replacement

    """


    # da[{"pos" : -2}] = (da[{"pos" : -2}] + da[{"pos" : -3}])/2
    # da[{"pos" : 1}] = (da[{"pos" : 1}] + da[{"pos" : 2}])/2
    
    da[-2] = (da[-2] + da[-3])/2
    da[1] = (da[1] + da[2])/2

    return da
    


