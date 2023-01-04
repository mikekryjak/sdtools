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