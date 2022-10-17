#!/usr/bin/env python3

import argparse
import os

def clean(path_case):
    """
    "Deletes all files apart from BOUT.inp and BOUT.settings
    """
    
    # Save original directory and enter the case folder
    original_dir = os.getcwd()
    os.chdir(original_dir + os.path.sep + path_case)
    files_removed = []
    
    for file in os.listdir(os.getcwd()):
        if any(x in file for x in [".nc", "log", "restart", "kate-swp", "pid"]):
            if "BOUT" in file:
                files_removed.append(file)
                os.remove(file)
            

    if len(files_removed)>0:
        print(f"-> Case {path_case} cleaned, files removed: {files_removed}")
    else:
        print("Nothing to clean")
        
    # Get back to the original directory
    os.chdir(original_dir)

    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "SD1D options reader")

    # Mandatory argument
    parser.add_argument("case", type=str, help = "Clean this case")
    # Extract arguments
    args = parser.parse_args()

    case = args.case
    clean(case)
    