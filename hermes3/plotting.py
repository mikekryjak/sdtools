import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from hermes3.utils import *
from hermes3.named_selections import *


class Monitor():
    def __init__(self, case, windows, settings = None):
        
        self.settings = settings
        self.plot_settings = {"xmin":None, "xmax":None}
        
        if self.settings is not None:
            if "plot" in self.settings.keys():
                for key in self.settings["plot"].keys():
                    self.plot_settings[key] = self.settings["plot"][key]
        
        self.fig_size = 3
        
        self.case = case
        self.ds = self.case.ds
        # self.windows = np.array(windows)
        self.windows = windows
        num_rows = len(self.windows)
        
        self.noguards = case.select_region("all_noguards")
        self.core = case.select_region("core_noguards")
        self.sol = case.select_region("sol_noguards")
        
        self.c = ["navy", "deeppink", "teal", "darkorange"]
        
        for row_id in range(num_rows):
        
            row_windows = self.windows[row_id]
            num_windows = len(row_windows)
            fig, self.axes = plt.subplots(1, num_windows, figsize = (self.fig_size*num_windows*1.2, self.fig_size))
            fig.subplots_adjust(wspace = 0.4)
            
            if num_windows == 1:
                self.add_plot(self.axes, row_windows[0])
            else:
                for i, name in enumerate(row_windows):
                    self.add_plot(self.axes[i], name)
        
        
    def add_plot(self, ax, name):
        
        legend = True
        xformat = True
        
        if "cvode" in name:
            cvode = dict()
            
            for param in ["cvode_nfevals", "cvode_npevals", "cvode_nliters", "cvode_nniters", "cvode_nsteps", "ncalls", "cvode_num_fails", "cvode_nonlin_fails"]:
                if "cvode" in param:
                    param_name = param.split("cvode_")[1]
                else:
                    param_name = param
                
                cvode[param_name] = np.gradient(self.ds[param], self.ds.coords["t"])
            
        
        if name == "target_temp":
            targets = dict()
            for target in  ["outer_lower"]:
                targets[target] = Target(self.case, target)
                ax.plot(self.ds["t"], targets[target].peak_temperature, label = target, color = self.c[0]) 
                ax.set_ylabel("Target temp [eV]")
                ax.set_title("Target temp")
                ax.set_yscale("log")


        elif name == "density":
            self.core["Ne"].mean(["x", "theta"]).plot(ax = ax, label = "Ne core", ls = "--", c = self.c[0])
            self.core["Nd"].mean(["x", "theta"]).plot(ax = ax, label = "neut core", ls = "--", c = self.c[1])
            self.sol["Ne"].mean(["x", "theta"]).plot(ax = ax, label = "Ne sol", c = self.c[0])
            self.sol["Nd"].mean(["x", "theta"]).plot(ax = ax, label = "neut sol", c = self.c[1])
            ax.legend(fontsize=9, loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 2)
            # ax.set_yscale("log")
            
        elif name == "temperature":
            self.core["Td+"].mean(["x", "theta"]).plot(ax = ax, label = "Td+ core", ls = "--", c = self.c[0])
            self.core["Te"].mean(["x", "theta"]).plot(ax = ax, label = "Te core", ls = "--", c = self.c[1])
            self.sol["Td+"].mean(["x", "theta"]).plot(ax = ax, label = "Td+ sol", c = self.c[0])
            self.sol["Te"].mean(["x", "theta"]).plot(ax = ax, label = "Te sol", c = self.c[1])
            ax.set_yscale("log")
            ax.legend(fontsize=9, loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 2)
            
        elif name == "radiation":
            (self.core["Rd+_ex"].mean(["x", "theta"])*-1).plot(ax = ax, label = "core", c = self.c[0])
            (self.sol["Rd+_ex"].mean(["x", "theta"])*-1).plot(ax = ax, label = "sol", c = self.c[1])
            ax.legend(fontsize=9, loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 2)

        elif name == "ionisation":
            self.core["Sd+_iz"].mean(["x", "theta"]).plot(ax = ax, label = "core", c = self.c[0])
            self.sol["Sd+_iz"].mean(["x", "theta"]).plot(ax = ax, label = "sol", c = self.c[1])
            ax.legend(fontsize=9, loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 2)
            
        elif name == "recombination":
            abs(self.core["Sd+_rec"].mean(["x", "theta"])).plot(ax = ax, label = "core", c = self.c[0])
            abs(self.sol["Sd+_rec"].mean(["x", "theta"])).plot(ax = ax, label = "sol", c = self.c[1])
            ax.legend(fontsize=9, loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 2)
            
        elif name == "core_temp_gradient":
            """ 
            Difference in temp on OMP between x indices a and b
            """
            omp = self.case.select_region("outer_midplane_a").sel(t=slice(None,None))

            a = 2
            b = 15
            
            (omp["Td+"].isel(x=a) - omp["Td+"].isel(x=b)).plot.line(ax = ax, x="t", label = "Ions", c = self.c[0])
            (omp["Te"].isel(x=a) - omp["Te"].isel(x=b)).plot.line(ax = ax, x="t", label = "Electrons", c = self.c[1])
            
            ax.set_ylabel("Temperature [eV]")
            ax.set_title(f"OMP Core temperature gradient between x={a} and {b}")
            ax.legend(fontsize=9, loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 2)
            
        elif name == "core_dens_gradient":
            """ 
            Difference in dens on OMP between x indices a and b
            """
            omp = self.case.select_region("outer_midplane_a").sel(t=slice(None,None))

            a = 2
            b = 15
            
            (omp["Ne"].isel(x=a) - omp["Ne"].isel(x=b)).plot.line(ax = ax, x="t", c = self.c[0], label = "Ne")
            
            ax.set_title(f"OMP Core plasma density gradient between x={a} and {b}")
            
        elif name == "cvode_order":
            ax.plot(self.ds.coords["t"], self.ds.data_vars["cvode_last_order"].values, label = "last_order", lw = 1, c = self.c[0])
            
        elif name == "cvode_evals":
            ax.plot(self.ds.coords["t"], cvode["nsteps"], c = self.c[0], label = "nsteps")
            ax.plot(self.ds.coords["t"], cvode["nfevals"], c = self.c[1], label = "nfevals")
            ax.plot(self.ds.coords["t"], cvode["npevals"], c = self.c[2], label = "npevals")
            ax.plot(self.ds.coords["t"], cvode["nliters"], c = self.c[3], label = "nliters")
            ax.set_yscale("log")
            ax.legend(fontsize=9, loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 2)

        elif name == "cvode_fails":
       
            ax.plot(self.ds.coords["t"], cvode["num_fails"], c = self.c[0], label = "num_fails")
            ax.plot(self.ds.coords["t"], cvode["nonlin_fails"], c = self.c[1], label = "nonlin_fails")
            ax.set_yscale("linear")
            
        elif name == "cvode_last_step":
            ax.plot(self.ds.coords["t"], self.ds["cvode_last_step"], c = self.c[0])
            
        elif name == "cvode_ncalls_per_second":
            # Per second of time simulated
            ncalls_per_timestep = (self.ds["ncalls"].data[0:-1]/self.ds.coords["t"].diff("t"))
            ax.plot(self.ds.coords["t"][0:-1], ncalls_per_timestep, c = self.c[0], lw = 1, markersize=1, marker = "o")
            
        elif name == "cvode_ncalls_per_step":
            ncalls_per_step = (cvode["nfevals"] + cvode["npevals"]) / cvode["nsteps"]
            ax.plot(self.ds.coords["t"], ncalls_per_step, c = self.c[0], lw = 1, markersize=1, marker = "o")
            
        elif name == "cvode_linear_per_newton":
            ax.plot(self.ds.coords["t"], np.divide(self.ds["cvode_nliters"], self.ds["cvode_nniters"]), c = self.c[0], lw = 1, markersize=1, marker = "o")
            
        elif name == "cvode_precon_per_newton":
            ax.plot(self.ds.coords["t"], self.ds["cvode_npevals"]/self.ds["cvode_nliters"], c = self.c[0], lw = 1, markersize=1, marker = "o")
        
        elif name == "cvode_fails_per_step":
            ax.plot(self.ds.coords["t"], (self.ds["cvode_num_fails"] + self.ds["cvode_nonlin_fails"])/self.ds["cvode_nsteps"], c = self.c[0], lw = 1, markersize=1, marker = "o")
        
        elif name == "cvode_fails_per_second":
            # Per second of time simulated
            ax.plot(self.ds.coords["t"], cvode["num_fails"], c = self.c[0], lw = 1, markersize=1, marker = "o")
            
        elif name == "cvode_stab_lims":
            # Per second of time simulated
            ax.plot(self.ds.coords["t"], self.ds["cvode_stab_lims"], c = self.c[0], lw = 1, markersize=1, marker = "o")
        
        else:
            raise Exception(f"Plot {name} not available")

        ax.set_title(name)
        if self.plot_settings["xmin"] is not None:
            ax.set_xlim(left = self.plot_settings["xmin"])
            
        if self.plot_settings["xmax"] is not None:
            ax.set_xlim(right = self.plot_settings["xmax"])
        # ax.set_xlim((self.plot_settings["xmin"], self.plot_settings["xmax"]))

        
    
        # ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1e}"))
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation = 0)
        ax.grid(which="both", alpha = 0.3)
        
        
        
class Monitor2D():
    """ 
    Plot monitor windows in the pattern [[1, 2, 3], [4, 5, 6]] or [[1]]
    mode is grid or pcolor
    """
    def __init__(self, case, mode, windows, settings = None):
        
        # Catch if user supplies just a [param] instead of [[param]]
        # if len(windows[0])==1 and isinstance(windows, list):
        #     windows = [windows]
            
        self.input_settings = settings
        
        # Defaults:
        self.settings = {"all": {"xlim":(None, None), "ylim":(None, None), 
                                 "figure_aspect":0.9, "wspace_modifier":1, 
                                 "view":None, "dpi":100}}
        
        
        self.fig_size = 3.5
        self.mode = mode
        self.case = case
        self.ds = self.case.ds
        

        self.capture_setting_inputs("all")

            
            
        # Set figure layout
        if mode == "grid":
            self.fig_height = self.fig_size * self.settings["all"]["figure_aspect"]
            self.wspace = 0.25
            
        elif "history" in mode:
            self.fig_height = 0.8 * self.fig_size * self.settings["all"]["figure_aspect"]
            self.wspace = 0.3
            
        elif mode == "pcolor":
            
            if self.settings["all"]["view"] == "lower_divertor":
                
                self.settings["all"]["figure_aspect"] = 0.5
                self.settings["all"]["ylim"] = (None,0)
                self.wspace = 0.15
                
            else:
                self.wspace = 0.2
                
            self.fig_height = 1.8 * self.fig_size * self.settings["all"]["figure_aspect"]
            
        
            
        
        print(self.settings["all"])
        self.windows = windows
        num_rows = len(self.windows)
        
        self.c = mike_cmap()
        
        for row_id in range(num_rows):
        
            row_windows = self.windows[row_id]
            num_windows = len(row_windows)
            fig, self.axes = plt.subplots(1, num_windows, dpi = self.settings["all"]["dpi"],
                                          figsize = (self.fig_size*num_windows, self.fig_height))
            fig.subplots_adjust(wspace = self.wspace)
            
            if num_windows == 1:
                self.add_plot(self.axes, row_windows[0])
            else:
                for i, name in enumerate(row_windows):
                    self.add_plot(self.axes[i], name)
        
    def capture_setting_inputs(self, plot):
        """
        Take settings from inputs and pass them to the correct plot
        """
        if self.input_settings is not None:
            if plot in self.input_settings.keys():
                for setting in self.input_settings[plot].keys():
                    self.settings[plot][setting] = self.input_settings[plot][setting]
                        
    def add_plot(self, ax, name):
        
        # Default settings
        self.settings[name] = {
            "log":True, "vmin":self.ds[name].min().values, "vmax":self.ds[name].max().values,
            }
        if "history" in self.mode:
            self.settings[name]["log"] = False
        # Modify through inputs
        self.capture_setting_inputs(name)
        

        settings = self.settings[name]
        
        if settings["vmin"] == None:
            settings["vmin"] = self.ds[name].min().values
            
        if settings["vmax"] == None:
            settings["vmax"] = self.ds[name].max().values
        
        meta = self.ds.metadata
        
        
        if self.mode == "grid":
        
            norm = create_norm(logscale = settings["log"], norm = None, vmin = settings["vmin"], vmax = settings["vmax"])
            abs(self.ds[name].isel(t=-1)).plot(ax = ax, cmap = "Spectral_r", cbar_kwargs={"label":""}, vmin=settings["vmin"], vmax=settings["vmax"], norm = norm)
            ax.set_title(name)

            ax.set_ylabel(""); ax.set_xlabel("")
            ax.tick_params(axis="x", labelrotation = 0)
            ax.grid(which="both", alpha = 0.3)
            
            ax.hlines(meta["ixseps1"], self.ds["theta"][0], self.ds["theta"][-1], colors = "k", ls = "--", lw = 1)
            
            
        elif self.mode == "pcolor":
            abs(self.ds[name].isel(t=-1)).bout.pcolormesh(ax = ax, cmap = "Spectral_r", logscale=settings["log"], vmin=settings["vmin"], vmax=settings["vmax"])#, cbar_kwargs={"label":""})
            ax.set_title(name)

            ax.set_ylabel(""); ax.set_xlabel("")
            ax.tick_params(axis="x", labelrotation = 0)
            ax.grid(which="both", alpha = 0.3)
            # [ax.vlines(meta[x], self.ds["x"][0], self.ds["x"][-1], colors = "k") for x in ["jyseps1_1", "jyseps1_2", "jyseps2_1", "jyseps2_2"]]
            
        
            
        elif self.mode == "omp_history":
            norm = create_norm(logscale = settings["log"], norm = None, vmin = settings["vmin"], vmax = settings["vmax"])
            self.case.select_region("outer_midplane_a")[name].plot(x = "t", ax = ax, cmap = "Spectral_r", norm = norm, cbar_kwargs={"label":""})
            ax.set_title(f"OMP {name}")
            
        elif self.mode == "target_history":
            norm = create_norm(logscale = settings["log"], norm = None, vmin = settings["vmin"], vmax = settings["vmax"])
            self.case.select_region("outer_lower_target")[name].plot(x = "t", ax = ax, cmap = "Spectral_r", norm = norm, cbar_kwargs={"label":""})
            ax.set_title(f"Target {name}")
            
            
        elif self.mode == "field_line_history":
            norm = create_norm(logscale = settings["log"], norm = None, vmin = settings["vmin"], vmax = settings["vmax"])
            self.case.select_custom_sol_ring(1, "outer_lower")[name].plot(x = "t", ax = ax, cmap = "Spectral_r", norm = norm, cbar_kwargs={"label":""})
            ax.set_title(f"1st SOL ring {name}")
            
        else:
            raise Exception(f"Mode {self.mode} not supported")
            
        if self.settings["all"]["ylim"] != (None, None):
            ax.set_ylim(self.settings["all"]["ylim"])
            
        if self.settings["all"]["xlim"] != (None, None):
            ax.set_xlim(self.settings["all"]["xlim"])

            
            
def plot_ddt(case, smoothing = 1, dpi = 120, volume_weighted = True, ylims = (None,None), xlims = (None,None)):
    """
    RMS of all the ddt parameters, which are convergence metrics.
    Inputs:
    smoothing: moving average period used for plot smoothing (default = 20. 1 is no smoothing)
    volume_weighted: weigh the ddt by cell volume
    """
    # Find parameters (species dependent)
    list_params = []

    for var in case.ds.data_vars:
        if "ddt" in var and not any([x in var for x in []]):
            list_params.append(var)
    list_params.sort()
    
    # Account for case if not enough timesteps for smoothing
    if len(case.ds.coords["t"]) < smoothing:
        smoothing = len(case.ds.coords) / 10

    res = dict()
    ma = dict()

    for param in list_params:

        if volume_weighted:
            res[param] = (case.ds[param] * case.dv) / np.sum(case.dv)    # Cell volume weighted
        else:
            res[param] = case.ds[param]
        res[param] = np.sqrt(np.mean(res[param]**2, axis = (1,2)))    # Root mean square
        res[param] = np.convolve(res[param], np.ones(smoothing), "same")    # Moving average with window = smoothing

    fig, ax = plt.subplots(figsize = (5,4), dpi = dpi)

    for param in list_params:
        ax.plot(case.ds.coords["t"], res[param], label = param, lw = 1)
        
    ax.set_yscale("log")
    ax.grid(which = "major", lw = 1)
    ax.grid(which = "minor", lw = 1, alpha = 0.3)
    ax.legend(loc = "upper left", bbox_to_anchor=(1,1))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cell weighted residual RMS [-]")
    ax.set_title(f"Residual plot: {case.name}")
    
    if ylims != (None,None):
        ax.set_ylim(ylims)
    if xlims != (None,None):
        ax.set_xlim(xlims)

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
                
    
def plot_selection(case, selection, dpi = 100):
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
    fig, axes = plt.subplots(1,3, figsize=(12,5), dpi = dpi, gridspec_kw={'width_ratios': [2.5, 1, 2]})
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
    if self.ds.metadata["topology"] != "single-null":
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
    
    if self.ds.metadata["topology"] != "single-null":
        ax.plot(self.Rxy[self.ixseps2,:], self.Zxy[self.ixseps2,:], label = "ixseps2", color = self.colors[5], alpha = 0.7, lw = 2, ls=":")

    if xlim != (None,None):
        ax.set_xlim(xlim)
    if ylim != (None,None):
        ax.set_ylim(ylim)
        
        
def create_norm(logscale, norm, vmin, vmax):
    if logscale:
        if norm is not None:
            raise ValueError(
                "norm and logscale cannot both be passed at the same time."
            )
        if vmin * vmax > 0:
            # vmin and vmax have the same sign, so can use standard log-scale
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            # vmin and vmax have opposite signs, so use symmetrical logarithmic scale
            if not isinstance(logscale, bool):
                linear_scale = logscale
            else:
                linear_scale = 1.0e-5
            linear_threshold = min(abs(vmin), abs(vmax)) * linear_scale
            norm = mpl.colors.SymLogNorm(linear_threshold, vmin=vmin, vmax=vmax)
    elif norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return norm