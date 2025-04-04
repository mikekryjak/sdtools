import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
# plt.style.use(r"C:\Users\mikek\OneDrive\Project\python-packages\sdtools\general\mike.mplstyle")


mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["mathtext.default"] = "regular"

mpl.rcParams["axes.grid"] = False
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["axes.formatter.limits"] = (-3,3)
mpl.rcParams["axes.formatter.useoffset"] = False
# mpl.rcParams["axes.prop_cycle"] = cycler("color", ["teal", "darkorange", "firebrick", "limegreen", "darkorchid", "deeppink", 
#                                                    "#1E90FF", "forestgreen", "#4169e1", "darkgoldenrod", "#9370db", "crimson", "#2f4f4f" ])

# newcycle = ["#0C9481", "#EC9714", "#18569A", "#EC6D14"]
# newcycle = ["#02A992", "#105BAF", "#FF9C04", "#FF6C04"]
# newcycle = ["#02A992", "#0B76AA", "#FF8A04", "#FF6C04"]  # Hue 206, dist 15deg
# newcycle = ["#43B7C2", "#024B79", "#FFAD48", "#BA5800"]
# newcycle = ["#43B7C2", "#FFAD48", "#024B79", "#BA5800"]

# mpl.rcParams["axes.prop_cycle"] = cycler("color", newcycle + ["darkorchid", "limegreen", "deeppink", 
#                                                    "#1E90FF", "forestgreen", "#4169e1", "darkgoldenrod", "#9370db", "crimson", "#2f4f4f" ] + default_cycle)
# mpl.rcParams["axes.prop_cycle"] = cycler("color", ["teal", "darkorange", "#fb8072", "#4daf4a", "#1f78b4", "darkorchid", "deeppink"])

default_cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']

def change_colors(theme):
    
    if theme == "default":
        mpl.rcParams["axes.prop_cycle"] = cycler("color", default_cycler)
    elif theme == "standard":
        mpl.rcParams["axes.prop_cycle"] = cycler("color", ["teal", "darkorange", "firebrick", "limegreen", "mediumblue", "darkorchid", "deeppink", "#1E90FF", "forestgreen", "#4169e1", "darkgoldenrod", "#9370db", "crimson", "#2f4f4f"] + default_cycler)
    elif theme == "pastel":
        mpl.rcParams["axes.prop_cycle"] = cycler("color", ["teal", "darkorange", "#fb8072", "#4daf4a", "#1f78b4", "darkorchid", "deeppink"] + default_cycler)

change_colors("standard")


mpl.rcParams["axes.autolimit_mode"] = "data"

mpl.rcParams["axes.xmargin"] = 0.1   # Margins for auto lims
mpl.rcParams["axes.ymargin"] = 0.1



# Make minor ticks invisible
mpl.rcParams["ytick.minor.width"] = 0
mpl.rcParams["xtick.minor.width"] = 0

# Ticks facing in and width 1
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.major.width"] = 1
mpl.rcParams["ytick.major.width"] = 1

mpl.rcParams["grid.linestyle"] = "-"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["grid.color"] = "lightgrey"
mpl.rcParams["grid.alpha"] = 0.5

mpl.rcParams["legend.loc"] = "best"
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["legend.markerscale"] = 1.0
mpl.rcParams['legend.borderaxespad'] = 0.1
# mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.facecolor'] = "white"

# mpl.rcParams["axes.spines.right"] = False
# mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["figure.figsize"] = (5,4)
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["figure.facecolor"] = "white"



mpl.rcParams['savefig.bbox'] = 'tight'
# mpl.rcParams['savefig.dpi'] = 500
mpl.rcParams['savefig.transparent'] = False

# mpl.rcParams["font.family"] = "sans-serif"
# mpl.rcParams["font.sans-serif"] = ["Helvetica"]

mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Tahoma', 'DejaVu Sans',
                               'Lucida Grande', 'Verdana']


        
        
"""
Here are seven colors that should work well with your existing colors:

Dodger Blue
Forest Green
Royal Blue
Dark Goldenrod
Medium Purple
Crimson
Dark Slate Gray

"""

class colordb():
    def __init__(self):
        pass
    
    def cycle(self):
        plt.rcParams['axes.prop_cycle'].by_key()['color']
        
    def colors_from_cmap(self, cmap, N):
        mpl.colormaps[cmap](np.linspace(0, 1, N))
