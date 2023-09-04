import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

onedrive_path = onedrive_path = str(os.getcwd()).split("OneDrive")[0] + "OneDrive"
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\sdtools"))

import sdtools.general.plotstyle

def read_trim(path, plot = True, debug = False, return_df = True):
    """
    Parses TRIM data from eirene
    
    Inputs
    -----
    path : str
        Path to the TRIM output file
    plot : bool
        Plot the results
    debug : bool
        Print file line by line with what the code thinks it is
    return_df : bool
        Return dataframe with results
    """
    
    with open(path, "r") as f:
        lines = f.readlines()
        
    ### Parse into list of dictionaries
    header_id = 0
    table_id = -1
    rawtables = []

    for i, line in enumerate(lines):
        lines[i] = line.replace("\n", "")
        
        if len(line.split()) > 0 and line.split()[0] == "1.":
            if debug is True: print("header >>", line)
            header_id = i
            table_id += 1
            rawtables.append({"header":line, "energy_dist":[], "polar":[], "azimuthal":[]})
            
        elif header_id != 0 and i == header_id + 2:
            if debug is True: print("nrgdst >>", line)
            rawtables[table_id]["energy_dist"].append(line)
            
        elif header_id != 0 and i > header_id + 3 and i < header_id + 3+6:
            if debug is True: print("polar  >>", line)
            rawtables[table_id]["polar"].append(line)
            
        elif header_id != 0 and i > header_id + 3+6 and i < header_id + 3+6+1+25:
            if debug is True: print("azimu  >>", line)
            rawtables[table_id]["azimuthal"].append(line)
            
        else:
            if debug is True: print("***** >>", line)
            
    
    ### Collect into dataframe
    df = pd.DataFrame()
    for i, table in enumerate(rawtables):
            header = table["header"].split()
            energy_dist = [float(x) for x in table["energy_dist"][0].split()]
            
            df.loc[i, "Ei"] = float(header[4])   # Incident energy
            df.loc[i, "angle"] = float(header[5])   # Angle of incidence
            df.loc[i, "Rf"] = float(header[6])    # Fast reflected fraction
            df.loc[i, "Es"] = np.mean(energy_dist)   # Scattered energy

    df["alpha"] = df["Es"] / df["Ei"]   # Reflected energy fraction
    
    print("*"*40)
    print("NOTE: 0 DEGREES IS NORMAL INCIDENCE")
    print("*"*40)
    
    if plot is True:
        fig, axes = plt.subplots(1,2, figsize = (12,4), dpi = 120)

        list_angle = df.groupby("angle").first().index.values

        ax = axes[0]
        for angle in list_angle:
            sel = df.query(f"angle == {angle}")
            axes[0].plot(sel["Ei"], sel["alpha"], label = f"${angle:.0f}\degree$")
            axes[1].plot(sel["Ei"], sel["Rf"])
            

        axes[0].set_ylabel("Energy reflection coefficient")
        axes[0].set_title(r"Energy reflection coefficient $\alpha$")

        axes[1].set_ylabel("Fast reflection fraction")
        axes[1].set_title(r"Fast reflection fraction $R_{f}$")

        for ax in axes:
            ax.set_xlabel("Incident energy [eV]")
            ax.set_xscale("log")
        fig.legend(ncols = 7, bbox_to_anchor = (0.5, -0.05), loc = "upper center")
        
    if return_df is True:
        return df