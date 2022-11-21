#!/usr/bin/env python3

import argparse
import os
import shutil
from clean import *
from clone import *
from read_opt import *
from set_opt import *

def clone_power(case, preserve = False, double = False):
    """
    Read a case and its power flux. If it's 1MW, clone it into a 6MW.
    If it's 6MW, clone it into a 1MW case. Rename accordingly.
    Double flag doubles powers.
    """
        
    path_file = case + os.path.sep + "BOUT.inp"
    
    power = read_opt(case, quiet = True)["p:powerflux"]
    power_suffix = case.split("-")[-1]
    
    # This would be bad.
    if power_suffix != power.split("e")[0]:
        print("WARNING: SUFFIX DOES NOT MATCH SETTING. EXITING")
        quit()
            
    # Figure out what power we need
    if double == False:
        if power == "1e6":
            new_power = "6e6"
        if power == "6e6":
            new_power = "1e6"
            
    elif double == True:
        if power == "2e6":
            new_power = "12e6"
        if power == "12e6":
            new_power = "2e6"
        
    # Figure out new name
    new_suffix = new_power.split("e")[0]
    new_name = case.replace(f'-{power_suffix}', f'-{new_suffix}')
    
    # Clone and set
    clone(case, new_name, preserve = True)
    set_opt(new_name, "p:powerflux", new_power, preserve = preserve)
    
    print("Power clone operation successful")
    
    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "SD1D options setter")
    parser.add_argument("case", type=str, help = "Clone this case")
    parser.add_argument("--preserve", action="store_true", help = "Preserve results after changing input?")
    parser.add_argument("--double", action="store_true", help = "Double powers?")

    # Extract arguments and call function
    args = parser.parse_args()
    clone_power(args.case, preserve = args.preserve)
    
    

