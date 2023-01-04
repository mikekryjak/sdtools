import numpy as np
import matplotlib.pyplot as plt

def plot_residuals(self):
    """
    Scaled residual calculation based on ANSYS Fluent
    From 26.13-19 in https://www.afs.enea.it/project/neptunius/docs/fluent/html/ug/node812.htm       
    - Take RMS of rate of change in each parameter over whole domain
    - Normalise by maximum value of this parameter within first 5 iterations
    - Plots parameters corresponding to equations solved (density, momentum, pressure)
    """
    # Find parameters (species dependent)
    list_params = ["Ne", "Pe"]

    for var in self.ds.data_vars:
        if "NV" in var and not any([x in var for x in ["S", ")", "_"]]):
            list_params.append(var)
        if "P" in var and not any([x in var for x in ["S", ")", "_", "e"]]):
            list_params.append(var)
    list_params.sort()

    
    res = dict()

    for param in list_params:

        res[param] = np.diff(self.ds[param], axis = 0) # Rate of change
        res[param] = np.sqrt(res[param]**2) # RMS of rate of change
        res[param] = np.mean(res[param], axis = (1,2))
        res[param] = res[param] / np.max(res[param][:4]) # Normalise by max in first 5 iterations

    fig, ax = plt.subplots(dpi = 100)

    for param in list_params:
        ax.plot(res[param], label = param)
    ax.set_yscale("log")
    ax.grid(which = "major", lw = 1)
    ax.grid(which = "minor", lw = 1, alpha = 0.3)
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalised residual")
    ax.set_title(f"Residual plot: {self.name}")
    
def plot_ddt(self, smoothing = 50, volume_weighted = True):
    """
    RMS of all the ddt parameters, which are convergence metrics.
    Inputs:
    smoothing: moving average period used for plot smoothing (default = 20. 1 is no smoothing)
    volume_weighted: weigh the ddt by cell volume
    """
    # Find parameters (species dependent)
    list_params = []

    for var in self.ds.data_vars:
        if "ddt" in var and not any([x in var for x in []]):
            list_params.append(var)
    list_params.sort()
    
    # Account for case if not enough timesteps for smoothing
    if len(self.ds.coords["t"]) < smoothing:
        smoothing = len(self.ds.coords) / 10

    res = dict()
    ma = dict()

    for param in list_params:

        if volume_weighted:
            res[param] = (self.ds[param] * self.dv) / np.sum(self.dv)    # Cell volume weighted
        else:
            res[param] = self.ds[param]
        res[param] = np.sqrt(np.mean(res[param]**2, axis = (1,2)))    # Root mean square
        res[param] = np.convolve(res[param], np.ones(smoothing), "same")    # Moving average with window = smoothing

    fig, ax = plt.subplots(figsize = (8,6), dpi = 100)

    for param in list_params:
        ax.plot(res[param], label = param, lw = 1)
        
    ax.set_yscale("log")
    ax.grid(which = "major", lw = 1)
    ax.grid(which = "minor", lw = 1, alpha = 0.3)
    ax.legend(loc = "upper left", bbox_to_anchor=(1,1))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalised residual")
    ax.set_title(f"Residual plot: {self.name}")

def plot_monitors(self, to_plot, what = ["mean", "max", "min"], ignore = []):
    """
    Plot time histories of parameters (density, pressure, or momentum)
    In each plot the solid line is the mean and dashed lines 
    represent the min/max at each timestep.
    Momentum is shown as an absolute value
    """

    list_params = []
    if to_plot == "pressure":
        for var in self.ds.data_vars:
            if "P" in var and not any([x in var for x in ignore+["S", ")", "_", ]]):
                list_params.append(var)
    elif to_plot == "density":
        for var in self.ds.data_vars:
            if "N" in var and not any([x in var for x in ignore+["S", ")", "_", "V"]]):
                list_params.append(var)
    elif to_plot == "momentum":
        for var in self.ds.data_vars:
            if "NV" in var and not any([x in var for x in ignore+["S", ")", "_"]]):
                list_params.append(var)
                
    else:
        list_params = to_plot

    list_params.sort()
    

    data = dict()

    for param in list_params:
        data[param] = dict()
        if "mean" in what:
            data[param]["mean"] = np.mean(self.ds[param], axis = (1,2))
        if "max" in what:
            data[param]["max"] = np.max(self.ds[param], axis = (1,2))
        if "min" in what:
            data[param]["min"] = np.min(self.ds[param], axis = (1,2))

        if to_plot == "momentum":
            for key in data[param]:
                data[param][key] = np.abs(data[param][key])

    colors = ["teal", "darkorange", "firebrick",  "limegreen", "magenta", "cyan", "navy"]
    fig, ax = plt.subplots(dpi = 100)

    for i, param in enumerate(list_params):
        if "mean" in what:
            ax.plot(data[param]["mean"], ls = "-", label = f"{param}", color = colors[i])
        if "max" in what:
            ax.plot(data[param]["max"], ls = ":",  color = colors[i])
        if "min" in what:
            ax.plot(data[param]["min"], ls = ":",  color = colors[i])

    ax.set_yscale("log")
    ax.grid(which = "major", lw = 1)
    ax.grid(which = "minor", lw = 1, alpha = 0.3)
    ax.legend(loc = "upper left", bbox_to_anchor=(1,1))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.set_title(f"{to_plot}: {self.name}")
    
    
def diagnose_cvode(self, lims = (0,0), scale = "log"):
    ds = self.ds

    fig, axes = plt.subplots(2,2, figsize = (8,6))

    ax = axes[0,0];  ax.set_yscale(scale)
    ax.plot(ds.coords["t"], ds.data_vars["cvode_nsteps"].values, label = "nsteps")
    ax.plot(ds.coords["t"], ds.data_vars["cvode_nfevals"].values, label = "nfevals")
    ax.plot(ds.coords["t"], ds.data_vars["cvode_npevals"].values, label = "npevals")
    ax.plot(ds.coords["t"], ds.data_vars["cvode_nliters"].values, label = "nliters")

    ax = axes[0,1]
    ax.plot(ds.coords["t"], ds.data_vars["cvode_last_order"].values, label = "last_order", lw = 1)

    ax = axes[1,0]; ax.set_yscale(scale)
    ax.plot(ds.coords["t"], ds.data_vars["cvode_num_fails"].values, label = "num_fails")
    ax.plot(ds.coords["t"], ds.data_vars["cvode_nonlin_fails"].values, label = "nonlin_fails")

    ax = axes[1,1]; ax.set_yscale(scale)
    ax.plot(ds.coords["t"], ds.data_vars["cvode_stab_lims"].values, label = "stab_lims")
    ax.plot(ds.coords["t"], ds.data_vars["cvode_stab_lims"].values, label = "last_step")

    for i in range(2):
        for j in range(2): 
            axes[i,j].grid()
            axes[i,j].legend()
            if lims != (0,0):
                axes[i,j].set_xlim(lims)
                
    
def plot_selection(case, selection):
    """ 
    Plot selected points on a R,Z grid
    X,Y grid doesn't work - need to fix it
    It originally generated a 2D array of X and Y indices so that you could
    easily get the coordinates of each point, and then sliced it by a slice object.
    Now I changed it to plot a ds's points over the base dataset, and need to regenerate
    the points somehow. Perhaps need to use something like meshgrid or something
    """
    self = case
    meta = self.ds.metadata

    # Region boundaries
    # ny = meta["ny"]     # Total ny cells (incl guard cells)
    # nx = meta["nx"]     # Total nx cells (excl guard cells)
    # Rxy = self.ds["R"].values    # R coordinate array
    # Zxy = self.ds["Z"].values    # Z coordinate array
    # MYG = meta["MYG"]

    # Array of radial (x) indices and of poloidal (y) indices in the style of Rxy, Zxy
    # x_idx = np.array([np.array(range(nx))] * int(ny + MYG * 4)).transpose()
    # y_idx = np.array([np.array(range(ny + MYG*4))] * int(nx))

    # Slice the X, Y and R, Z arrays and vectorise them for plotting
    # xselect = selection["x"].values.flatten()
    # yselect = np.where(ds["theta"].values == selection["theta"].values)    # Theta converted to index space
    rselect = selection["R"].values.flatten()
    zselect = selection["Z"].values.flatten()

    # Plot
    fig, axes = plt.subplots(1,3, figsize=(12,5), dpi = 100, gridspec_kw={'width_ratios': [2.5, 1, 2]})
    fig.subplots_adjust(wspace=0.3)

    plot_xy_grid(case, axes[0])
    # axes[0].scatter(yselect, xselect, s = 4, c = "red", marker = "s", edgecolors = "darkorange", linewidths = 0.5)

    plot_rz_grid(case, axes[1])
    axes[1].scatter(rselect, zselect, s = 20, c = "red", marker = "s", edgecolors = "darkorange", linewidths = 1, zorder = 100)

    plot_rz_grid(case, axes[2], ylim=(-1,-0.25))
    axes[2].scatter(rselect, zselect, s = 20, c = "red", marker = "s", edgecolors = "darkorange", linewidths = 1, zorder = 100)
    
def plot_xy_grid(case, ax):
    self = case
    ax.set_title("X, Y index space")
    ax.scatter(self.yflat, self.xflat, s = 1, c = "grey")
    ax.plot([self.yflat[self.j1_1g]]*np.ones_like(self.xflat), self.xflat, label = "j1_1g",   color = self.colors[0])
    ax.plot([self.yflat[self.j1_2g]]*np.ones_like(self.xflat), self.xflat, label = "j1_2g", color = self.colors[1])
    ax.plot([self.yflat[self.j2_1g]]*np.ones_like(self.xflat), self.xflat, label = "j2_1g",   color = self.colors[2])
    ax.plot([self.yflat[self.j2_2g]]*np.ones_like(self.xflat), self.xflat, label = "j2_2g", color = self.colors[3])
    ax.plot(self.yflat, [self.yflat[self.ixseps1]]*np.ones_like(self.yflat), label = "ixseps1", color = self.colors[4])
    ax.plot(self.yflat, [self.yflat[self.ixseps2]]*np.ones_like(self.yflat), label = "ixseps1", color = self.colors[5], ls=":")
    ax.legend(loc = "upper center", bbox_to_anchor = (0.5,-0.1), ncol = 3)
    ax.set_xlabel("Y index (incl. guards)")
    ax.set_ylabel("X index (excl. guards)")

def plot_rz_grid(case, ax, xlim = (None,None), ylim = (None,None)):
    self = case
    ax.set_title("R, Z space")
    ax.scatter(self.rflat, self.zflat, s = 0.1, c = "black")
    ax.set_axisbelow(True)
    ax.grid()
    ax.plot(self.Rxy[:,self.j1_1g], self.Zxy[:,self.j1_1g], label = "j1_1g",     color = self.colors[0], alpha = 0.7)
    ax.plot(self.Rxy[:,self.j1_2g], self.Zxy[:,self.j1_2g], label = "j1_2g", color = self.colors[1], alpha = 0.7)
    ax.plot(self.Rxy[:,self.j2_1g], self.Zxy[:,self.j2_1g], label = "j2_1g",     color = self.colors[2], alpha = 0.7)
    ax.plot(self.Rxy[:,self.j2_2g], self.Zxy[:,self.j2_2g], label = "j2_2g", color = self.colors[3], alpha = 0.7)
    ax.plot(self.Rxy[self.ixseps1,:], self.Zxy[self.ixseps1,:], label = "ixseps1", color = self.colors[4], alpha = 0.7, lw = 2)
    ax.plot(self.Rxy[self.ixseps2,:], self.Zxy[self.ixseps2,:], label = "ixseps2", color = self.colors[5], alpha = 0.7, lw = 2, ls=":")

    if xlim != (None,None):
        ax.set_xlim(xlim)
    if ylim != (None,None):
        ax.set_ylim(ylim)