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

def show_log(key, n):
        
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
        fnkey = "*"
    else:
        fnkey = "*" + str(key) + "*"
        
    if n == None:
        n = 10
    else:
        n = int(n)
        
    folders = os.listdir(os.getcwd())
    path = os.getcwd()
    statuses = []
    cases = []
    mtimes = []

    for folder in folders:
        
        # Check it's not a file
        if fnmatch.fnmatch(folder, fnkey) and "." not in folder:

            path_folder = path + os.path.sep + folder

            files = os.listdir(path_folder)

            found_log = False
            found_inp = False
            
            # Make sure we have results and input files
            for file in files:
                if "dmp" in file:
                    found_dmp = True
                if ".inp" in file:
                    found_inp = True
                    
            if found_dmp == True and found_inp == True:
                print("_____________________________________________________________________")
                print(f"\nFound case {folder}--------------------------------------------------")
                print("_____________________________________________________________________")
                path_log = os.path.join(path_folder, "BOUT.log.0")
                
                with open(path_log) as f:
                    logfile = f.readlines()
                    
                for i in range(n):
                    i = n - i
                    print(logfile[-i])
                
                #print(logfile[-n:])

        
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Prints case status")
    parser.add_argument("key", type=str, nargs = "?", help = "Only return cases with this in name")
    parser.add_argument("nlines", type=str, nargs = "?", help = "Number of lines to read")
    
    # Extract arguments
    args = parser.parse_args()
    key = args.key

    show_log(key, n = args.nlines)
