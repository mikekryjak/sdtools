#!/usr/bin/env python3

from boutdata.data import BoutData
import os
import sys
import getopt
from optparse import OptionParser
import argparse
from datetime import datetime
import pandas as pd
import fnmatch

def is_finished(key):
        
    """
    -> is_finished(path, key, quiet = False)
    Will print status of all cases with names matching key in path
    Statuses:
    Finished - Current timestep matches intended
    Error - can't read boutdata
    Not finished - Current timestep doesn't match intended
    Not started - Input file exists but no dump file
    Missing input file - No input file
    """
    if key == None:
        key = "*"
    key = str(key)
    folders = os.listdir(os.getcwd())
    path = os.getcwd()
    statuses = []
    cases = []
    mtimes = []

    for folder in folders:
        
        # Check it's not a file
        if fnmatch.fnmatch(folder, key) and "." not in folder:

            path_folder = path + os.path.sep + folder

            files = os.listdir(path_folder)

            found_dmp = False
            found_inp = False
            boutdata_ok = False
            
            # Make sure we have results and input files
            for file in files:
                if "dmp" in file:
                    found_dmp = True
                if ".inp" in file:
                    found_inp = True

            if found_dmp == True:
                try:
                    data = BoutData(path + os.path.sep + folder)
                    boutdata_ok = True
                except:
                    print("Tried to read a file that's not a case: {}".format(folder))

            if boutdata_ok == True:
                if data["options"]["timestep"] * data["options"]["nout"] == data["outputs"]["tt"]:
                    status = "Finished"
                else:
                    status = "Not finished"
                    
            if boutdata_ok == False:
                status = "Error"
            if found_dmp == False:
                status = "Not started"
            if found_inp == False:
                status = "Missing input file"
                
            # Find date modified
            mtime = os.path.getmtime(folder)
            mtime = datetime.fromtimestamp(mtime)
           
            mtimes.append(mtime)
            cases.append(folder)
            statuses.append(status)
        
    if __name__ == "__main__":
        # Create df to sort by time modified      
        out = pd.DataFrame()
        out["mtime"] = mtimes
        out["case"] = cases
        out["status"] = statuses
        out = out.sort_values(by="mtime", ascending = False)

        for i in range(len(out)):
            row = out.iloc[i]

            print(f'{row["mtime"].strftime("%d/%m/%Y %H:%M")} || {row["case"]} || {row["status"]}')
    
    # If called as a function, return just the status.        
    else:
        if len(statuses) > 1:
            print("Found multiple files, please provide unique key")
        else:
            return(status)
        
        
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Prints case status")
    parser.add_argument("key", type=str, nargs = "?", help = "Only return cases with this in name")

    # Extract arguments
    args = parser.parse_args()
    key = args.key

    is_finished(key)
