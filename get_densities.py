#!/usr/bin/env python3
from boutdata.collect import collect
from boutdata.data import BoutData
import argparse
import fnmatch
import os


def get_densities(casepath):
    
    """
    Get line averaged plasma and neutral densities from case
    """
    data = BoutData(casepath)
    Nnorm = data["outputs"]["Nnorm"]
    Tnorm = data["outputs"]["Tnorm"]
    
    Nn = data["outputs"]["Nn"][-1, 0, :, 0] * Nnorm
    Ne = data["outputs"]["Ne"][-1, 0, :, 0] * Nnorm
    dy = data["outputs"]["dy"][0, :]
    J = data["outputs"]["J"][0, :]
    
    dV = dy * J

    Ne_avg = sum(Ne * dV) / sum(dy) 
    Nn_avg = sum(Nn * dV) / sum(dy)
    Ntot_avg = Ne_avg + Nn_avg
    Nu = Ne[0]

    print(f"\n>>> {casepath}")
    print(f"Ne: {Ne_avg:.3E} || Nn: {Nn_avg:.3E} || Total: {Ntot_avg:.3E}")
    print(f"-------------------------------------------------------------")
    print(f"Nu: {Nu:.3E}\n")


#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Line averaged density calculator")
    parser.add_argument("key", type=str, help = "Post-process cases matching this key")

    # Extract arguments and call function
    args = parser.parse_args()
      
        # Fnmatch needs * for wildcards.
    fnsearch = f'*{args.key}*'
    
    cwd = os.getcwd()
    sep = os.path.sep
    
    # If not a file, and if fnmatch found it, and if it has an input file... collect
    to_read = []
    for folder in os.listdir(cwd):
        if "." not in folder and fnmatch.fnmatch(folder, fnsearch) and "BOUT.dmp.0.nc" in os.listdir(cwd+sep+folder):
            to_read.append(folder)
            
    for case in to_read:
        get_densities(case)
