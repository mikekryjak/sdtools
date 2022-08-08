#!/usr/bin/env python3

import argparse
import os
import shutil
from clone import *
from read_opt import *
from set_opt import *


def make_scan(case, mode, overwrite = False):
    """
    Takes one case and creates density scans to hardcoded density
    New cases are renamed and a suffix -x is appended to indicate unfinished case
    Cases are cleaned and if overwrite=True, new cases overwrite any old ones
    Mode is either double_power or density
    """
    # case = "c1-r1-2"
    # path_case = os.getcwd()+"\\cases\\"+case
    
    path_case = os.path.join(os.getcwd(), case)
    
    prefix = case.split("-")[0]
    caseid = case.split("-")[1]
    suffix = case.split("-")[2]

    if mode == "density":
        intend_param = float(suffix) * 1e19
        scan = [1, 2, 3, 5, 7, 10]
        case_param = float(read_opt(path_case, quiet = True)["ne:function"]) * 1e20
        
    elif mode == "double_power":
        intend_param = float(suffix) * 1e6 * 2
        scan = [1, 2, 3, 4, 5, 6]
        case_param = float(read_opt(path_case, quiet = True)["p:powerflux"]) * 1e6 * 2
        
    if float(suffix) in scan:
        scan.remove(float(suffix))

    print("-----------------------------------------------------------")
    print(f"Cloning case {case} onto {mode} scan {scan}")
    if overwrite:
        print("Overwrite set to true!\n")

    if intend_param != case_param:
        print(f"Case {case} parameter mismatch. Found: {case_param:.1E} || Case name implies: {intend_param:.1E}")
        print("Correcting param to match case name.")
        
        if mode == "density":
            set_opt(path_case, "ne:function", intend_param/1e20)
            
        elif mode == "double_power":
            set_opt(path_case, "p:powerflux", "{:.1E}".format(intend_param).replace("E+0", "e"))

    new_names = []

    for i, param in enumerate(scan):
        
        new_names.append(prefix + "-" + caseid + "-" + str(int(param)))

    for i, new_case in enumerate(new_names):
        clone(path_case, new_case, overwrite = True, preserve = False)

        path_new_case = os.path.dirname(path_case) + os.path.sep + new_case
        
        if mode == "density":
            set_opt(path_new_case, "ne:function", scan[i]/10)
        if mode == "double_power":
            set_opt(path_new_case, "p:powerflux", "{:.1E}".format(scan[i]*1e6*2).replace("E+0", "e"))

        print("Created new case {}\n".format(path_new_case))
        
        
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "SD1D scan creator")
    parser.add_argument("case", type=str, help = "Clone this case")
    parser.add_argument("mode", type=str, help = "Either double_power or density")
    parser.add_argument("--overwrite", action="store_true", help = "Overwrite?")

    # Extract arguments and call function
    args = parser.parse_args()
    make_scan(args.case, args.mode, overwrite = args.overwrite)
