#!/usr/bin/env python3

import argparse
import os
import shutil
from clean import *
from read_opt import *
import fnmatch

def set_opt(case, opt, new_value, preserve = False):
    """
    Read BOUT.inp and replace a parameter value with a new one.
    Format for param is category:param, e.g. mesh:length_xpt.
    """
        
    path_file = case + os.path.sep + "BOUT.inp"
    
    settings = read_opt(case, quiet = True)
    old_value = settings[opt]
    
    with open(path_file) as f:
        lines = f.readlines()
    lines_new = lines.copy()

    category = ""
    replaced = False
    print("-> Opened {}".format(path_file))
    for i, line in enumerate(lines):
        if "[" in line[0]:
        # If category
            category = line.split("[")[1].split("]")[0].lower() 
            
        # If the correct line, replace.
        # Prints done without \n for formatting reasons
        found = False
        
        # If one of those options without a category (which are hardcoded):
        if opt in ["timestep", "nout"] and opt in line and old_value in line:
            found = True
            
        # If one of the other options:
        elif category == opt.split(":")[0] and opt.split(":")[1] in line and old_value in line:
            found = True
            
        if found == True:
            print("Old line:", line.replace("\n",""))
            line = line.replace(old_value, str(new_value))
            replaced = True
            print("New line:", line.replace("\n",""))
            lines_new[i] = line
            

    if replaced == False:
        print("Parameter not found!")
            
    # Write file
    with open(path_file, "w") as f:
        f.writelines(lines_new)
        
    if preserve == False:
        print("Case written and results deleted")
    if preserve == True:
        print("Case written, results preserved")

    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "SD1D options setter")
    parser.add_argument("key", type=str, help = "Modify cases with this in name")
    parser.add_argument("opt", type=str, help = "Modify this option in category:option format")
    parser.add_argument("new_value", type=str, help = "New value to set")
    parser.add_argument("--preserve", action="store_true", help = "Preserve results after changing input?")

    # Extract arguments and call function
    args = parser.parse_args()
    key = str(args.key)
    fnmatch_key = "*" + key + "*"
    
    cwd = os.getcwd()
    sep = os.path.sep
    
    to_set = []
    for folder in os.listdir(cwd):
        if "." not in folder and fnmatch.fnmatch(folder, fnmatch_key) and "BOUT.inp" in os.listdir(cwd+sep+folder):
            to_set.append(folder)
            
    to_set.sort()

    if len(to_set) > 0:
        print(f"-> Apply {args.opt}={args.new_value} to:")
        [print(x) for x in to_set]
        answer = input("-> Confirm y/n:")
        
        if answer == "y":
            for case in to_set:
                set_opt(case, args.opt, args.new_value, preserve = args.preserve)
                
        elif answer == "n":
            print("Exiting")
            quit()
            
        else:
            print("You were supposed to type y or n. Exiting")
            quit()            
                
    else:
        print("Found no cases matching '{}'".format(args.key))
    
    
    
    

