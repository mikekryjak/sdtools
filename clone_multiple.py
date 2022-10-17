#!/usr/bin/env python3

import argparse
import os
import shutil
from clean import *
from clone import *
import fnmatch

"""
Clone multiple cases 
clone_multiple.py key  new --preserve --overwrite
key: operate on cases containing key. Key will be replaced with "new"
new: "key" will be replaced with this string.
--preserve: will refrain from deleting result files.
--overwrite: will overwrite files if they already exist.
"""
    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Mass case clone")
    parser.add_argument("key", type=str, help = "Clone from this case")
    parser.add_argument("new", type=str, help = "Replace key with this string")
    parser.add_argument("--clean", action="store_true", help = "Clean result files?")
    parser.add_argument("--overwrite", action="store_true", help = "Overwrite new case?")

    # Extract arguments
    args = parser.parse_args()
    key = str(args.key)
    new = args.new
    preserve = not args.clean
    overwrite = args.overwrite
    
    # Fnmatch needs * for wildcards.
    fnsearch = f'*{key}*'
    
    cwd = os.getcwd()
    sep = os.path.sep
    
    # If not a file, and if fnmatch found it, and if it has an input file... collect
    to_set = []
    for folder in os.listdir(cwd):
        if fnmatch.fnmatch(folder, fnsearch) and "BOUT.inp" in os.listdir(cwd+sep+folder):
            to_set.append(folder)
            
    if to_set == []:
        print("No cases found matching {}".format(key))
    else:
        print("-> Clone source -> clone destination")
        
    new_cases = []
        # Make new case name and clone
        
    for case in to_set:
        left = case.split(key)[0]
        right = case.split(key)[1]
        new_case = left + new + right
        new_cases.append(new_case)
        print(f"{case} - > {new_case}")
                    
    answer = input("Continue? y/n")
       
        
    if answer == "y":
        for i, case in enumerate(to_set):
            print("Creating case..", end = "")
            print(f"{case}...",end = "")
            clone(case, new_cases[i], overwrite = overwrite, preserve = preserve)
            print("-> Multiple clone operation completed")
        
    else:
        print("Exiting")
        quit()
    

    

