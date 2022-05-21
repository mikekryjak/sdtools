#!/usr/bin/env python3

import argparse
import os
import shutil
from clean import *
from read_opt import *

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
        if category == opt.split(":")[0] and opt.split(":")[1] in line and old_value in line:
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
    parser.add_argument("case", type=str, help = "Modify this case")
    parser.add_argument("opt", type=str, help = "Modify this option in category:option format")
    parser.add_argument("new_value", type=str, help = "New value to set")
    parser.add_argument("--preserve", action="store_true", help = "Preserve results after changing input?")

    # Extract arguments and call function
    args = parser.parse_args()
    set_opt(args.case, args.opt, args.new_value, preserve = args.preserve)
    
    

