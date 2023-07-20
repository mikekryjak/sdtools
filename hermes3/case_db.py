from pathlib import Path
import os

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
            