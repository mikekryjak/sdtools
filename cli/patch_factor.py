#!/usr/bin/env python3

import argparse
import os
import shutil
import fnmatch
from boutdata.restart import scalevar

"""
Scale a variable by a factor in the restart files
Using the Boutdata.restart.scalevar function
"""

#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "Restart file variable scaler")
    parser.add_argument("case", type=str, help = "Modify this case")
    parser.add_argument("variable", type=str, help = "Variable to scale")
    parser.add_argument("factor", type=float, help = "Scale factor")
    
    args = parser.parse_args()
    
    print(f">>> Scaling {args.variable} in {args.case} by factor of {args.factor}")
    scalevar(args.variable, args.factor, args.case)
    print("--> Complete")
    