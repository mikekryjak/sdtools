import platform

system = platform.system()



def read_opt(path_case, opts = None, group = None):
# Read BOUT.inp file and parse
# to produce dict of settings

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
                category = line.split("[")[1].split("]")[0] 
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
    if opts != None and group == None:
                        
        if type(opts) == list:
            for opt in opts:
                if opt in settings.keys():
                    print(f"{opt}: {settings[opt]}")
                else:
                    print("{}: Option not found".format(option))

        else:
            if opts in settings.keys():
                print(f"{opts}: {settings[opts]}")
            else:
                print("{}: Option not found".format(option))

    # Prints all settings for a group (e.g. "solver" or "sd1d")
    elif opts == None and group != None:
        if group in categories:
            for opt in settings.keys():
                if group in opt:
                    print(f"{opt}: {settings[opt]}")
        else:
            print(f"Group {group} not found")
        
                
    else:
        return settings
    

def set_opt(path_case, opt, new_value):
# Read BOUT.inp and replace a parameter value with a new one.
# Format for param is category:param, e.g. mesh:length_xpt.

    if system == "Windows":
        path_file = path_case + r"\\BOUT.inp"
    if system == "Linux":
        path_file = path_case + r"/BOUT.inp"
    
    settings = read_opt(path_case)
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
            category = line.split("[")[1].split("]")[0] 

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
    
    
def clone(path_case, new_case, f=False):
# Clones a case in path_case into new_case
# f is for Force, or overwrite

    if new_case in os.listdir():
        print(f"Case {new_case} already exists!")
        if f == True:
            print(f"Force enabled, deleting {new_case}")
            shutil.rmtree(new_case)
        else:
            print("-> Exiting")
        
    path_root = os.path.dirname(path_case)
    shutil.copytree(path_case, new_case)
    print(f"-> Copied case {path_case} into {new_case}")
    
    
def clean(path_case):
# Deletes all files apart from BOUT.inp and BOUT.settings
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
        print(f"Files removed: {files_removed}")
    else:
        print("Nothing to clean")
        
    # Get back to the original directory
    os.chdir(original_dir)
    
    

