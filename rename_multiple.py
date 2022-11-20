#!/usr/bin/env python3

import argparse
import os
import shutil
from clean import *
from clone import *
import fnmatch

"""
Rename multiple cases
clone_multiple.py key  new --preserve --overwrite
key: operate on cases containing key. Key will be replaced with "new"
new: "key" will be replaced with this string.
"""
    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Mass case clone")
    parser.add_argument("key", type=str, help = "Clone from this case")
    parser.add_argument("new", type=str, help = "Replace key with this string")

    # Extract arguments
    args = parser.parse_args()
    key = str(args.key)
    new = args.new

    
    # Fnmatch needs * for wildcards.
    fnsearch = f"*{key}*"
    
    cwd = os.getcwd()
    sep = os.path.sep
    
    # If not a file, and if fnmatch found it, and if it has an input file... collect
    to_set = []
    for folder in os.listdir(cwd):
        if fnmatch.fnmatch(folder, fnsearch):
            to_set.append(folder)
            
    if to_set == []:
        print("No cases found matching {}".format(key))
    else:
        print("-> Old name -> New name")
        
        new_cases = []
            
        for case in to_set:
            new_case = case.replace(key, new)
            new_cases.append(new_case)
            print(f"{case} - > {new_case}")
                        
        answer = input("Continue? y/n")
        
            
        if answer == "y":
            for i, case in enumerate(to_set):
                print("Renaming..", end = "")
                print(f"{case} -> {new_cases[i]}")
                os.rename(case, new_cases[i])
            print("-> Rename completed")
            
        else:
            print("Exiting")
            quit()
        

    

