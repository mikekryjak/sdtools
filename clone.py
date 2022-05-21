#!/usr/bin/env python3

import argparse
import os
import shutil
from clean import *

def clone(case, new_case, overwrite = False, preserve = False):
    """
    Clones a case in path_case into new_case
    f is for Force, or overwrite
    c is for clean, remove all files but .inp and .settings from new cases
    """
    
    path_root = os.getcwd()
    path_new_case = path_root + os.path.sep + new_case

    if new_case in os.listdir(path_root):
        print(f"Case {new_case} already exists!")

        if overwrite == True:
            print(f"-> Force enabled, deleting {new_case}")
            shutil.rmtree(path_new_case)

        else:
            print("-> Exiting")
            quit()
        
    shutil.copytree(case, path_new_case)
    print(f"-> Copied case {case} into {new_case}")
    
    if preserve == False:
        print("opps")
        clean(new_case)

    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "SD1D options reader")
    parser.add_argument("case", type=str, help = "Clone from this case")
    parser.add_argument("new_case", type=str, help = "Clone to this case")
    parser.add_argument("--preserve", action="store_true", help = "Preserve result files?")
    parser.add_argument("--f", action="store_true", help = "Overwrite new case?")

    # Extract arguments
    args = parser.parse_args()
    case = args.case
    new_case = args.new_case
    preserve = args.preserve
    overwrite = args.f

    clone(case, new_case, overwrite = overwrite, preserve = preserve)

