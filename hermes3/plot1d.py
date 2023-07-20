import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from hermes3.utils import *
from hermes3.named_selections import *
from matplotlib.widgets import RangeSlider, TextBox, Slider
import animatplot as amp


def animate_with_reference(ds, params, auxparam):
    
    

    dpi = 100

    wspace = 0.2
    # params = ["NVd+", "Sd+_iz", "Sd+_rec"]
    params = ["NVd+", "Sd+_iz", "Sd+_rec"]
    # params = ["Sd+_"]
    # auxparam = ds["ncalls"]
    auxparam = ((ds["Nd+"]+ds["Nd"]) * ds["dv"]).sum("pos")
    t = ds.coords["t"].values
    num_plots = len(params)



    ds = ds.isel(pos=slice(1,-1))

    fig = plt.figure(dpi=dpi)
    fig.set_figheight(4)
    fig.set_figwidth(num_plots*6)
    fig.subplots_adjust(bottom = 0.2)

    # Plot grid
    gs0a = mpl.gridspec.GridSpec(
                                    ncols=num_plots+1, nrows=1,
                                    wspace = wspace
                                    )

    axes = [None] * num_plots
    data = [None] * num_plots
    blocks = [None] * num_plots

    timeline = amp.Timeline(t, units = "s", fps = 10)


    pos = ds["pos"].values

    for i, param in enumerate(params):
        data[i] = ds[param].values

    # Make scanning line
    vline = np.repeat(np.array([[min(auxparam.values), max(auxparam.values)]]), len(t), axis = 0)


    for i, param in enumerate(params):
        # All plots after the first one share x and y axes
        if i == 0:
            axes[i] = fig.add_subplot(gs0a[i])
        else:
            axes[i] = fig.add_subplot(gs0a[i], sharex=axes[0])
            
        blocks[i] = amp.blocks.Line(pos, ds[param].values, ax = axes[i], color = "black", marker = "o", markersize = 1)
        # axes[i].set_xlim(77,78.2)
        axes[i].set_title(param)
        axes[i].grid()
        

    # Aux plot and vline
    axes.append(fig.add_subplot(gs0a[len(params)]))
    axes[-1].plot(t, auxparam, c = "darkslategrey")
    axes[-1].grid()

    # Line is two points, these are the X coordinates for those two points for all times
    x = np.tile(t,  (2,1)).transpose()
    blocks.append(amp.blocks.Line(x, vline, ax = axes[-1], t_axis=0, color = "deeppink"))

        
    anim = amp.Animation(blocks, timeline)
    anim.controls({"text":"TIME", "color":"darkorange", "valfmt":"%1.3e"})
    aux_plot_idx = num_plots
        



