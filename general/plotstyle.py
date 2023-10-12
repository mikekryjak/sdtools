import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
# plt.style.use(r"C:\Users\mikek\OneDrive\Project\python-packages\sdtools\general\mike.mplstyle")


mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["mathtext.default"] = "regular"

mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.grid.which"] = "both"
mpl.rcParams["axes.formatter.limits"] = (-3,3)
mpl.rcParams["axes.formatter.useoffset"] = False
mpl.rcParams["axes.prop_cycle"] = cycler("color", ["teal", "darkorange", "firebrick", "mediumblue", "limegreen", "darkorchid", "deeppink", 
                                                   "#1E90FF", "forestgreen", "#4169e1", "darkgoldenrod", "#9370db", "crimson", "#2f4f4f" ])
# mpl.rcParams["axes.prop_cycle"] = cycler("color", ["teal", "darkorange", "#fb8072", "#4daf4a", "#1f78b4", "darkorchid", "deeppink"])
mpl.rcParams["axes.autolimit_mode"] = "data"
mpl.rcParams["axes.xmargin"] = 0.1
mpl.rcParams["axes.ymargin"] = 0.1

mpl.rcParams["xtick.major.width"] = 1
mpl.rcParams["ytick.major.width"] = 1

mpl.rcParams["grid.linestyle"] = "-"
mpl.rcParams["grid.linewidth"] = 0.3
mpl.rcParams["grid.alpha"] = 0.15

mpl.rcParams["legend.loc"] = "best"
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["legend.markerscale"] = 2.0
mpl.rcParams["legend.framealpha"] = 0.3

mpl.rcParams["figure.figsize"] = (6,5)
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["figure.facecolor"] = "white"

def change_colors(theme):
    if theme == "default":
        mpl.rcParams["axes.prop_cycle"] = cycler("color", ["teal", "darkorange", "firebrick", "limegreen", "mediumblue", "darkorchid", "deeppink"])
    elif theme == "pastel":
        mpl.rcParams["axes.prop_cycle"] = cycler("color", ["teal", "darkorange", "#fb8072", "#4daf4a", "#1f78b4", "darkorchid", "deeppink"])
        
        
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