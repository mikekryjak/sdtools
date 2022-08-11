#!/usr/bin/env python3

import argparse
import numpy as np
import os
import shutil
from boutdata.data import BoutData
import fnmatch

"""
Clone multiple cases 
clone_multiple.py key  new --preserve --overwrite
key: operate on cases containing key. Key will be replaced with "new"
new: "key" will be replaced with this string.
--preserve: will refrain from deleting result files.
--overwrite: will overwrite files if they already exist.
"""

def time_stats(casepath):
    out = dict()
    out["wtime"] = sum(BoutData(casepath)["outputs"]["wtime"])/3600
    out["wtime_all"] = BoutData(casepath)["outputs"]["wtime"]
    out["wtime_avg"] = np.mean(out["wtime_all"])
    out["wtime_std"] = np.std(out["wtime_all"])
    return out
    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Runtime stats")
    parser.add_argument("key", type=str, help = "Show time stats for cases matching this key")

    # Extract arguments
    args = parser.parse_args()
    key = str(args.key)
    
    # Fnmatch needs * for wildcards.
    fnsearch = f'*{key}*'
    
    cwd = os.getcwd()
    sep = os.path.sep
    
    # If not a file, and if fnmatch found it, and if it has an input file... collect
    to_set = []
    for folder in os.listdir(cwd):
        casepath = os.path.join(cwd,folder)
        if fnmatch.fnmatch(folder, fnsearch) and "BOUT.inp" in os.listdir(casepath):
            out = time_stats(casepath)
            print(f"{folder} -> Wall time: {out['wtime']:.2f}hrs || Standard deviation: {out['wtime_std']:.2f}")


    

