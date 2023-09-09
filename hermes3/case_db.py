from pathlib import Path
import os
from boutdata.squashoutput import squashoutput
from datetime import datetime as dt
from boututils.options import BOUTOptions
from hermes3.load import *

class CaseDB():
    """ 
    Find all simulations and grids in the provided directories
    store their paths in dictionaries casepaths and gridpaths.
    Defaults to a hardcoded relative OneDrive path
    """
    def __init__(self, case_dir = None,
                       grid_dir = None):
        
        
        # Set default paths
        onedrive_path = onedrive_path = str(os.getcwd()).split("OneDrive")[0] + "OneDrive"
        case_dir = os.path.join(onedrive_path, r"Project\collab\tech\cases") if case_dir == None else case_dir
        grid_dir = os.path.join(onedrive_path, r"Project\collab\tech\grid") if grid_dir == None else grid_dir
        
        self.casepaths = dict()
        self.gridpaths = dict()
        self.cases = []
        self.grids = []

        for input_file in Path(case_dir).rglob('BOUT.inp'):
            self.casepaths[input_file.parent.name] = input_file.parent
            self.cases.append(input_file.parts[-2])

        for grid_file in Path(grid_dir).rglob('*.nc'):
            self.gridpaths[grid_file.name] = grid_file
            self.grids.append(grid_file.parts[-2])
            
    def get_grid_path(self, casename):
        """
        Find path of grid based on case name
        """
        
        if casename not in self.casepaths.keys():
            raise Exception(f"Case {casename} not found in database")
        
        casepath = self.casepaths[casename]
        options = BOUTOptions()
        options.read_inp(casepath)

        gridname = options.mesh["file"]

        if gridname in self.gridpaths.keys():
            gridfilepath = self.gridpaths[gridname]
        else:
            raise Exception(f"Grid {gridname} not found in database")
        
        return gridfilepath
    
    def load_case_2D(self, 
                    casename,
                    verbose = False, 
                    squeeze = True, 
                    unnormalise_geom = True,
                    unnormalise = True,
                    use_squash = False):

        casepath = self.casepaths[casename]
        gridfilepath = self.get_grid_path(casename)
            
        return Load.case_2D(
                    casepath = casepath,
                    gridfilepath = gridfilepath,
                    verbose = verbose,
                    squeeze = squeeze,
                    unnormalise_geom = unnormalise_geom,
                    unnormalise = unnormalise,
                    use_squash = use_squash
                )

def squash(casepath, verbose = True, force = False):
    """
    Checks if squashed file exists. If it doesn't, or if it's older than the dmp files, 
    then it creates a new squash file.
    
    Inputs
    ------
    Casepath is the path to the case directory
    verbose gives you extra info prints
    force always recreates squash file
    """
    
    datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
    inputfilepath = os.path.join(casepath, "BOUT.inp")
    squashfilepath = os.path.join(casepath, "BOUT.squash.nc") # Squashoutput hardcoded to this filename

    recreate = True if force is True else False   # Override to recreate squashoutput
    squash_exists = False
    
    if verbose is True: print(f"- Looking for squash file")
        
    if "BOUT.squash.nc" in os.listdir(casepath):  # Squash file found?
        
        squash_exists = True
        
        squash_date = os.path.getmtime(squashfilepath)
        dmp_date = os.path.getmtime(os.path.join(casepath, "BOUT.dmp.0.nc"))
        
        squash_date_string = dt.strftime(dt.fromtimestamp(squash_date), r"%m/%d/%Y, %H:%M:%S")
        dmp_date_string = dt.strftime(dt.fromtimestamp(dmp_date), r"%m/%d/%Y, %H:%M:%S")
        
        if verbose is True: print(f"- Squash file found. squash date {squash_date_string}, dmp file date {dmp_date_string}") 
        
        if dmp_date > squash_date:   #Recreate if squashoutput is too old
            recreate = True
            print(f"- dmp files are newer than the squash file! Recreating...") 
            
    else:
        if verbose is True: print(f"- Squashoutput file not found, creating...")
        recreate = True
        

    if recreate is True:
        
        if squash_exists is True:  # Squashoutput will not overwrite, so we delete the file first
            os.remove(squashfilepath)
            
        squashoutput(
            datadir = casepath,
            outputname = squashfilepath,
            xguards = True,   # Take all xguards
            yguards = "include_upper",  # Take all yguards (yes, confusing name)
            parallel = False,   # Seems broken atm
            quiet = verbose
        )
        
        if verbose is True: print(f"- Done")
            