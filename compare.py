#!/usr/bin/env python3

import argparse
import os
from read_opt import *

"""
Compare two cases
compare.py case1 case2 
"""
    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Case comparison")
    parser.add_argument("case1", type=str, help = "First case to compare")
    parser.add_argument("case2", type=str, help = "Second case to compare")
    args = parser.parse_args()

    # Get paths
    cwd = os.getcwd()
    case1 = args.case1
    case2 = args.case2

    path1 = os.path.join(cwd, case1)
    path2 = os.path.join(cwd, case2)

    settings1 = read_opt(path1, quiet = True)
    settings2 = read_opt(path2, quiet = True)
    set1 = set(settings1.items())
    set2 = set(settings2.items())

    diff1 = dict(set1 - set2)
    diff2 = dict(set2 - set1)

    all_keys = set(list(diff1.keys()) + list(diff2.keys()))


    # common_keys = set(diff1).intersection(set(diff2))

    if settings1 == settings2:
        print(f"Cases {case1} and {case2} have identical settings")
    else:
        print("\n>>> Found differences in settings:------------------")
        for key in all_keys:
            print(f"\n>{key}")

            if key in settings1.keys():
                print(f"{case1}: {settings1[key]}")
            else:
                print(f"{case1}: NOT PRESENT")

            if key in settings2.keys():
                print(f"{case2}: {settings2[key]}")
            else:
                print(f"{case2}: NOT PRESENT")

    print("")
