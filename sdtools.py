#!/usr/bin/env python3

import os
import platform
from optparse import OptionParser
import argparse
import shutil
from boutdata.data import BoutData


system = platform.system()




def read_opt(path_case, keys = None, quiet = False):
    """
    Read BOUT.inp file and parse
    to produce dict of settings
    """

    if system == "Windows":
        path_file = path_case + r"\\BOUT.inp"
    if system == "Linux":
        path_file = path_case + r"/BOUT.inp"
    
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
                        
    # Prints only specific settings, either specified as string or list of strings 
    if quiet == False:
        for opt in settings.keys():
            if keys != None and any(key in opt for key in keys):
                print(f"{opt}: {settings[opt]}")
            elif keys == []:
                print(f"{opt}: {settings[opt]}")
    else:
        return settings # Quiet mode suppresses print and returns the settings dict

def set_opt(path_case, opt, new_value):
    """
    Read BOUT.inp and replace a parameter value with a new one.
    Format for param is category:param, e.g. mesh:length_xpt.
    """

    if system == "Windows":
        path_file = path_case + r"\\BOUT.inp"
    if system == "Linux":
        path_file = path_case + r"/BOUT.inp"
    
    settings = read_opt(path_case, quiet = True)
    old_value = settings[opt]
    
    with open(path_file) as f:
        lines = f.readlines()
    lines_new = lines.copy()

    category = ""
    replaced = False
    print(">>>>>Opened {}".format(path_file))
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
    print("Case written successfully")
    
    
def clone(path_case, new_case, force = False, c=False):
    """
    Clones a case in path_case into new_case
    f is for Force, or overwrite
    c is for clean, remove all files but .inp and .settings from new cases
    """
    
    path_root = os.path.dirname(path_case)
    path_new_case = path_root + os.path.sep + new_case
    
    # print(os.listdir(path_root))
    if new_case in os.listdir(path_root):
        print(f"Case {new_case} already exists!")
        if force == True:
            print(f"Force enabled, deleting {new_case}")
            shutil.rmtree(path_new_case)

        else:
            print("-> Exiting")
        
    shutil.copytree(path_case, path_new_case)
    print(f"-> Copied case {path_case} into {new_case}")
    
    if c==True:
        clean(path_new_case)
    
    
def clean(path_case):
    """
    "Deletes all files apart from BOUT.inp and BOUT.settings
    """

    print(f"Cleaning case {path_case}....")
    
    # Save original directory and enter the case folder
    original_dir = os.getcwd()
    os.chdir(path_case)
    files_removed = []
    
    for file in os.listdir(os.getcwd()):
        if any(x in file for x in [".nc", "log", "restart"]):
            files_removed.append(file)
            os.remove(file)
            

    if len(files_removed)>0:
        print(f"Case cleaned, files removed: {files_removed}")
    else:
        print("Nothing to clean")
        
    # Get back to the original directory
    os.chdir(original_dir)
    
    
def make_scan(case, overwrite = False):
    """
    Takes one case and creates density scans to hardcoded density
    New cases are renamed and a suffix -x is appended to indicate unfinished case
    Cases are cleaned and if overwrite=True, new cases overwrite any old ones
    """
    # case = "c1-r1-2"
    # path_case = os.getcwd()+"\\cases\\"+case
    
    prefix = case.split("-")[0]
    caseid = case.split("-")[1]
    densid = case.split("-")[2]

    intend_dens = float(densid) * 1e19
    dens_scan = [1e19, 2e19, 3e19, 5e19, 7e19, 10e19]

    print("-----------------------------------------------------------")
    print(f"Cloning case {case} onto density scan {dens_scan}")
    if overwrite:
        print("Overwrite set to true!\n")

    case_dens = float(read_opt(path_case, quiet = True)["ne:function"]) * 1e20

    if intend_dens != case_dens:
        print(f"Case {case} density mismatch. Found: {case_dens:.1E} || Case name implies: {intend_dens:.1E}")
        print("Correcting density to match case name.")
        set_opt(path_case, "ne:function", intend_dens/1e20)

    if intend_dens in dens_scan:
        dens_scan.remove(intend_dens)

    new_names = []

    for i, dens in enumerate(dens_scan):
        new_names.append(prefix + "-" + caseid + "-" + str(int(dens/1e19)) + "-x")

    for i, new_case in enumerate(new_names):
        clone(path_case, new_case, force = overwrite, c = True)

        path_new_case = os.path.dirname(path_case) + os.path.sep + new_case
        set_opt(path_new_case, "ne:function", dens_scan[i]/1e20)

        print("Created new case {}\n".format(path_new_case))
        
def is_finished(case_path, quiet = False):
    """
    Returns true if case finished
    """
    data = BoutData(case_path)

    if data["options"]["timestep"] * data["options"]["nout"] == data["outputs"]["tt"]:
        is_finished = True
    else:
        is_finished = False
        
    if quiet == False:
        print(is_finished)
        
    return is_finished
    
    
# def time_stats(path_case, quiet = False):
# """
# Return time it took to run a case.
# Runtime is total time. wtime is array of time taken per iteration.
# """
#     out = dict()
#     out["runtime"] = sum(BoutData(path_case)["outputs"]["wtime"])/3600
#     out["wtime"] = BoutData(path_case)["outputs"]["wtime"]
#     out["wtime_avg"] = np.mean(out["wtime"])
#     out["wtime_std"] = np.std(out["wtime"])
    
#     if quiet == False:
#         print("{}: runtime = {}s || wtime = {} || 
    
#     return out
    
    
# Create the parser

if __name__ == '__main__':
    
    # Make parsers
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")
    p_opt_read = subparser.add_parser("read_opt")
    p_set_opt = subparser.add_parser("set_opt")
    p_clone = subparser.add_parser("clone")
    p_clean = subparser.add_parser("clean")
    p_make_scan = subparser.add_parser("make_scan")
    p_is_finished = subparser.add_parser("is_finished")

    # Add arguments to parsers
    p_opt_read.add_argument('-i', type=str, nargs="+", required = True, 
                            help="Read settings from case. First input is case folder, remaining are keys to search for in options") 
    p_set_opt.add_argument("-i", type=str, nargs=3, required = True, 
                           help = "Change setting in a case. --set_opt(case_folder, setting_name, new_value")
    
    p_clone.add_argument("-i", type=str, nargs=2, required = True, 
                         help = "Clone case. --clone(case_folder, new_name)")
    p_clone.add_argument( "-f", action="store_true", 
                         help = "Overwrite old case")
    p_clone.add_argument( "-c", action="store_true", 
                         help = "Clean new case")
    
    p_clean.add_argument("-i", type=str, nargs=1, required=True, 
                         help = "Removes all but input and settings files. clean(case_folder)")
    
    p_make_scan.add_argument("-i", type=str, nargs=1, required=True, 
                             help = "Clones case into a density scan. -i to provide case name")
    p_make_scan.add_argument("-f", action="store_true", 
                             help = "Overwrite old cases")
    
    p_is_finished.add_argument("-i", type=str, nargs=1, required = True, 
                         help = "Tell you if case finished")
    
    # Link parser arguments to functions
    args = parser.parse_args()

    if args.command == "read_opt":
        read_opt(args.i[0], args.i[1:])

    if args.command == "set_opt":
        set_opt(args.i[0], args.i[1], args.i[2])

    if args.command == "clone":
        clone(args.i[0], args.i[1], force = args.f, c = args.c)

    if args.command == "clean":
        clean(args.i[0])
        
    if args.command == "make_scan":
        make_scan(args.i[0], overwrite = args.f)
        
    if args.command == "is_finished":
        is_finished(args.i[0])


