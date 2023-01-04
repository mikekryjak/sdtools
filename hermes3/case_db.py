from pathlib import Path

class CaseDB():
    """ 
    Find all simulations and grids in the provided directories
    store their paths in dictionaries casepaths and gridpaths
    """
    def __init__(self, case_dir = r"C:\Users\mikek\OneDrive\Project\collab\tech\cases\st40",
                       grid_dir = r"C:\Users\mikek\OneDrive\Project\collab\tech\grid"):
        
        self.casepaths = dict()
        self.gridpaths = dict()

        for input_file in Path(case_dir).rglob('BOUT.inp'):
            self.casepaths[input_file.parent.name] = input_file.parent

        for grid_file in Path(grid_dir).rglob('*.nc'):
            self.gridpaths[grid_file.name] = grid_file