#!/usr/bin/env python3

import argparse
import os


def read_opt(path_case, key = None, quiet = False):
    """
    Read BOUT.inp file and parse
    to produce dict of settings
    """
    
    path_file = path_case + os.path.sep + "BOUT.inp"
    
    with open(path_file) as f:
        lines = f.readlines()

    settings = dict()
    category = ""
    categories = []

    for i, line in enumerate(lines):
        if line[0] not in ["#", "\n"]:


            if "[" in line[0]:
                # If category
                category = line.split("[")[1].split("]")[0].lower() 
                categories.append(category)

            else:
                # Isolate param name and delete space afterwards
                param = line.split("=")[0][:-1]

                # Get rid of comment and newline if one exists
                if "#" in line:
                    line = line.split("#")[0]
                if "\n" in line:
                    line = line.split("\n")[0]

                # Extract value:
                if "=" in line:             
                    value = line.split("=")[1].strip()

                    if category != "":
                        settings[f"{category.lower()}:{param.lower().strip()}"] = value
                    else:
                        settings[param] = value
                        
    if __name__ == "__main__":
    # Prints only specific settings, either specified as string or list of strings 
        for opt in settings.keys():
            if key != None and key in opt:
                print(f"{opt}: {settings[opt]}")
            elif key == None:
                print(f"{opt}: {settings[opt]}")
    else:
        return settings # Quiet mode suppresses print and returns the settings dict

    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "SD1D options reader")
    parser.add_argument("case_key", type=str, help = "Read cases with this string in folder name")
    parser.add_argument("opt_key", type=str, nargs="?", help = "Optional: read options with this string in setting name")
    # Extract arguments
    args = parser.parse_args()
    case_key = args.case_key
    opt_key = args.opt_key

    cwd = os.getcwd()
    sep = os.path.sep

    to_read = []
    for folder in os.listdir(cwd):

        # For all folders that match key but don't have dots in names (i.e. are files)
        if any(x in folder for x in [case_key]) and not any(x in folder for x in ["."]):

            # Make sure there is a BOUT input file in the dir, and if so then append it.
            case_dir = cwd + sep + folder
            if "BOUT.inp" in os.listdir(case_dir):
                to_read.append(folder)

    if len(to_read) == 0:
        print("No cases found matching {}".format(case_key))

    for case in to_read:
        print("{}---------".format(case))

        read_opt(case, key = opt_key)
        print("")
