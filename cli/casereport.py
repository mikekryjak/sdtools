# from boututils.datafile import DataFile
# from boutdata.collect import collect
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, pathlib
import argparse
# import platform
# import traceback
# import xarray
# import xbout
# import scipy
# import re


sys.path.append("/work/e281/e281/mkryjak/python-packages/sdtools")

# from hermes3.utils import *
# from gridtools.hypnotoad_tools import *
# from gridtools.b2_tools import *
# from gridtools.utils import *

from hermes3.fluxes import *
from hermes3.case_db import *
from hermes3.load import *
from hermes3.named_selections import *
from hermes3.plotting import *
from hermes3.grid_fields import *
from hermes3.accessors import *

# from code_comparison.code_comparison import *
# from code_comparison.viewer_2d import *


parser = argparse.ArgumentParser(description = "Perform report")
parser.add_argument("gridpath", type=str, help = "Grid to use")
parser.add_argument("casepath", type=str, help = "Case to run")


args = parser.parse_args()

x = np.linspace(1,10,10)
y = np.linspace(1,10,10)
plt.plot(x,y)
plt.savefig("report.png")

case = Load.case_2D(args.casepath, args.gridpath, double_load = False, keep_xboundaries = True, 
                    keep_yboundaries = True, unnormalise_geom = True)

Monitor(case, [["sep_ne", "sep_te", "target_temp","radiation"]], save = True)