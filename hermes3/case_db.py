from pathlib import Path

class CaseDB():
    """ 
    Find all simulations and grids in the provided directories
    store their paths in dictionaries casepaths and gridpaths
    """
    def __init__(self, case_dir = r"C:\Users\mikek\OneDrive\Project\collab\tech\cases",
                       grid_dir = r"C:\Users\mikek\OneDrive\Project\collab\tech\grid"):
        
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
            