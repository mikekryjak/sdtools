from pathlib import Path
import os
from boutdata.squashoutput import squashoutput
from datetime import datetime as dt
from boututils.options import BOUTOptions
from hermes3.load import Load



class CaseDB():
    """ 
    Find all simulations and grids in the provided directories
    store their paths in dictionaries casepaths and gridpaths.
    Defaults to a hardcoded relative OneDrive path.
    
    Inputs
    ------
    case_dir : path where cases are
    grid_dir : path where grids are 
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
                    **kwargs
                    ):

        casepath = self.casepaths[casename]
        gridfilepath = self.get_grid_path(casename)
            
        return Load.case_2D(
                    casepath = casepath,
                    gridfilepath = gridfilepath,
                    **kwargs
                )

