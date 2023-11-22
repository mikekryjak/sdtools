import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from hermes3.utils import *
from hermes3.named_selections import *
import general.plotstyle



class Monitor():
    def __init__(self, case, windows, settings = None, save = False):
        
        self.settings = settings
        self.plot_settings = {"xmin":None, "xmax":None}
        self.save = save
        
        if self.settings is not None:
            if "plot" in self.settings.keys():
                for key in self.settings["plot"].keys():
                    self.plot_settings[key] = self.settings["plot"][key]
        
        self.fig_size = 3
        
        self.case = case
        self.ds = self.case.ds
        m = self.ds.metadata
        # self.windows = np.array(windows)
        self.windows = windows
        num_rows = len(self.windows)
        
        self.noguards = case.ds.hermesm.select_region("all_noguards")
        self.core = case.ds.hermesm.select_region("core_noguards")
        self.sol = case.ds.hermesm.select_region("sol_noguards")
        self.omp = case.ds.hermesm.select_region("outer_midplane_a").isel(x = slice(m["MYG"], -m["MYG"]))
        
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
            
            if self.save is True:
                fig.savefig(f"{case.name}_monitor_{row_id}.png")
        
        
    def add_plot(self,  ax, name):
        
        legend = True
        xformat = True
        m = self.ds.metadata
        
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


        elif name == "avg_density":
            self.core["Ne"].mean(["x", "theta"]).plot(ax = ax, label = "Ne core", ls = "--", c = self.c[0])
            self.core["Nd"].mean(["x", "theta"]).plot(ax = ax, label = "neut core", ls = "--", c = self.c[1])
            self.sol["Ne"].mean(["x", "theta"]).plot(ax = ax, label = "Ne sol", c = self.c[0])
            self.sol["Nd"].mean(["x", "theta"]).plot(ax = ax, label = "neut sol", c = self.c[1])
            ax.legend(fontsize=9, loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 2)
            ax.set_yscale("log")
            
        elif name == "avg_temp":
            self.core["Td+"].mean(["x", "theta"]).plot(ax = ax, label = "Td+ core", ls = "--", c = self.c[0])
            self.core["Te"].mean(["x", "theta"]).plot(ax = ax, label = "Te core", ls = "--", c = self.c[1])
            self.sol["Td+"].mean(["x", "theta"]).plot(ax = ax, label = "Td+ sol", c = self.c[0])
            self.sol["Te"].mean(["x", "theta"]).plot(ax = ax, label = "Te sol", c = self.c[1])
            ax.set_yscale("log")
            ax.legend(fontsize=9, loc = "upper center", bbox_to_anchor = (0.5, 1.35), ncol = 2)
            
        elif name == "sep_ne":
            self.omp["Ne"].isel(x = m["ixseps1"]).plot(ax = ax, c = self.c[0])
            
        elif name == "sep_te":
            self.omp["Te"].isel(x = m["ixseps1"]).plot(ax = ax, c = self.c[0])
            
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
            omp = self.case.ds.hermesm.select_region("outer_midplane_a").sel(t=slice(None,None))

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
            omp = self.case.ds.hermesm.select_region("outer_midplane_a").sel(t=slice(None,None))

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
        
        elif name == "cumulative_linear_fails_per_second":
            # Per second of time simulated
            ax.plot(self.ds.coords["t"], self.ds["cvode_num_fails"], c = self.c[0], lw = 1, markersize=1, marker = "o")
            
        elif name == "cumulative_newton_fails_per_second":
            # Per second of time simulated
            ax.plot(self.ds.coords["t"], self.ds["cvode_nonlin_fails"], c = self.c[0], lw = 1, markersize=1, marker = "o")
            
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
                                 "view":None, "dpi":100, "clean_guards":True}}
        
        
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
            
        elif mode == "pcolor" or mode == "polygon":
            
            # Select final timestep if one not provided
            if "t" in self.ds.dims.keys():
                self.ds = self.ds.isel(t=-1)
            
            if self.settings["all"]["view"] == "lower_divertor":
                
                self.settings["all"]["figure_aspect"] = 0.5
                self.settings["all"]["ylim"] = (None,0)
                self.wspace = 0.15
                
            else:
                self.wspace = 0.2
                
            self.fig_height = 1.8 * self.fig_size * self.settings["all"]["figure_aspect"]
            

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
            "log":False, "vmin":self.ds[name].min().values, "vmax":self.ds[name].max().values,
            }
        if "history" in self.mode:
            self.settings[name]["log"] = False
        # Modify through inputs
        self.capture_setting_inputs(name)
        
        

        settings = self.settings[name]
        
        
        
        if self.settings["all"]["clean_guards"] is True:
            self.ds[name] = self.ds[name].hermesm.clean_guards()
        
        if settings["vmin"] == None:
            settings["vmin"] = self.ds[name].min().values
            
        if settings["vmax"] == None:
            settings["vmax"] = self.ds[name].max().values
        
        meta = self.ds.metadata
        
        
        if self.mode == "grid":
        
            norm = create_norm(logscale = settings["log"], norm = None, vmin = settings["vmin"], vmax = settings["vmax"])
            abs(self.ds[name]).plot(ax = ax, cmap = "Spectral_r", cbar_kwargs={"label":""}, vmin=settings["vmin"], vmax=settings["vmax"], norm = norm)
            ax.set_title(name)

            ax.set_ylabel(""); ax.set_xlabel("")
            ax.tick_params(axis="x", labelrotation = 0)
            ax.grid(which="both", alpha = 0.3)
            
            ax.hlines(meta["ixseps1"], self.ds["theta"][0], self.ds["theta"][-1], colors = "k", ls = "--", lw = 1)
            
            
        elif self.mode == "pcolor":
            
            data = abs(self.ds[name])
            # if self.settings["all"]["clean_guards"] is True:
            #     data = data.hermesm.clean_guards()
                
            data.bout.pcolormesh(ax = ax, cmap = "Spectral_r", logscale=settings["log"], vmin=settings["vmin"], vmax=settings["vmax"])#, cbar_kwargs={"label":""})
            ax.set_title(name)

            ax.set_ylabel(""); ax.set_xlabel("")
            ax.tick_params(axis="x", labelrotation = 0)
            ax.grid(which="both", alpha = 0.3)
            # [ax.vlines(meta[x], self.ds["x"][0], self.ds["x"][-1], colors = "k") for x in ["jyseps1_1", "jyseps1_2", "jyseps2_1", "jyseps2_2"]]
            
        elif self.mode == "polygon":
            data = abs(self.ds[name])
            # if self.settings["all"]["clean_guards"] is True:
            #     data = data.hermesm.clean_guards()
                
            data.bout.polygon(ax = ax, cmap = "Spectral_r", logscale=settings["log"], vmin=settings["vmin"], vmax=settings["vmax"])#, cbar_kwargs={"label":""})
            ax.set_title(name)

            ax.set_ylabel(""); ax.set_xlabel("")
            ax.tick_params(axis="x", labelrotation = 0)
            ax.grid(which="both", alpha = 0.3)
            
        elif self.mode == "omp_history":
            norm = create_norm(logscale = settings["log"], norm = None, vmin = settings["vmin"], vmax = settings["vmax"])
            self.case.ds.hermesm.select_region("outer_midplane_a")[name].plot(x = "t", ax = ax, cmap = "Spectral_r", norm = norm, cbar_kwargs={"label":""})
            ax.set_title(f"OMP {name}")
            
        elif self.mode == "target_history":
            norm = create_norm(logscale = settings["log"], norm = None, vmin = settings["vmin"], vmax = settings["vmax"])
            self.case.ds.hermesm.select_region("outer_lower_target")[name].plot(x = "t", ax = ax, cmap = "Spectral_r", norm = norm, cbar_kwargs={"label":""})
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

            
            
def plot_ddt(case, 
             smoothing = 1, 
             dpi = 120, 
             volume_weighted = True, 
             ylims = (None,None), 
             xlims = (None,None)):
    """
    RMS of all the ddt parameters, which are convergence metrics.
    Inputs:
    smoothing: moving average period used for plot smoothing (default = 20. 1 is no smoothing)
    volume_weighted: weigh the ddt by cell volume
    """
    
    if case.is_2d:
        ds = case.ds.hermesm.select_region("all_noguards")
    else:
        # No guard cells at all
        ds = case.ds.isel(pos=slice(2,-2))
    

        
    
    
    # Find parameters (species dependent)
    list_params = []

    for var in ds.data_vars:
        if "ddt" in var and not any([x in var for x in []]):
            list_params.append(var)
    list_params.sort()
    
    # Account for when if not enough timesteps for smoothing
    if len(ds.coords["t"]) < smoothing:
        smoothing = len(ds.coords) / 10

    res = dict()
    ma = dict()
    

    for param in list_params:

        if volume_weighted:
            res[param] = (ds[param] * ds.dv) / np.sum(ds.dv)    # Cell volume weighted
        else:
            res[param] = ds[param]
            
        if case.is_2d:
            res[param] = np.sqrt(np.mean(res[param]**2, axis = (1,2)))    # Root mean square
        else:
            res[param] = np.sqrt(np.mean(res[param]**2, axis = 1))    # Root mean square
        res[param] = np.convolve(res[param], np.ones(smoothing), "same")    # Moving average with window = smoothing

    fig, ax = plt.subplots(figsize = (5,4), dpi = dpi)
    fig.subplots_adjust(right=0.8)


    for param in list_params:
        ax.plot(ds.coords["t"], res[param], label = param, lw = 1)
        
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
                
    
def plot_selection(ds, selection, dpi = 100, rz_only = False, show_selection = True):
    """ 
    Plot selected points on a R,Z grid
    X,Y grid doesn't work - need to fix it
    It originally generated a 2D array of X and Y indices so that you could
    easily get the coordinates of each point, and then sliced it by a slice object.
    Now I changed it to plot a ds's points over the base dataset, and need to regenerate
    the points somehow. Perhaps need to use something like meshgrid or something
    """

    m = ds.metadata


    # Slice the X, Y and R, Z arrays and vectorise them for plotting
    xselect = selection["x"].values.flatten()
    yselect = selection["theta_idx"].values.flatten()
    rselect = selection["R"].values.flatten()
    zselect = selection["Z"].values.flatten()

    # Plot
    fig, axes = plt.subplots(
        1,2, figsize=(8,5), dpi = dpi, gridspec_kw={'width_ratios': [2, 1]},
        )
    fig.subplots_adjust(
        wspace=0.15,
        top = 0.75, bottom = 0.05, left = 0.10, right = 0.95)

    plot_xy_grid(ds, axes[0])
    if show_selection is True:
        for x in xselect:
            axes[0].scatter(yselect, [x]*len(yselect), s = 4, c = "red", marker = "s", edgecolors = "darkorange", linewidths = 0.5)

    plot_rz_grid(ds, axes[1])
    if show_selection is True:
        axes[1].scatter(rselect, zselect, s = 20, c = "red", marker = "s", edgecolors = "darkorange", linewidths = 1, zorder = 100)

    
    
def plot_xy_grid(ds, ax):

    m = ds.metadata
    yflat = ds["y_idx"].data.flatten()
    xflat = ds["x_idx"].data.flatten()
        
    ax.set_title("X, Y index space")
    ax.scatter(yflat, xflat, s = 1, c = "grey")
    ax.plot([yflat[m["j1_1g"]]]*np.ones_like(xflat), xflat, label = "j1_1g",   color = m["colors"][0])
    ax.plot([yflat[m["j1_2g"]]]*np.ones_like(xflat), xflat, label = "j1_2g", color = m["colors"][1])
    ax.plot([yflat[m["j2_1g"]]]*np.ones_like(xflat), xflat, label = "j2_1g",   color = m["colors"][2])
    ax.plot([yflat[m["j2_2g"]]]*np.ones_like(xflat), xflat, label = "j2_2g", color = m["colors"][3])
    ax.plot(yflat, [yflat[m["ixseps1"]]]*np.ones_like(yflat), label = "ixseps1", color = m["colors"][4])
    if m["topology"] != "single-null":
        ax.plot(yflat, [yflat[m["ixseps2"]]]*np.ones_like(yflat), label = "ixseps1", color = m["colors"][5], ls=":")
        
    ax.plot(yflat[m["ny_inner"]]*np.ones_like(xflat), xflat, label = "ny_inner", color = m["colors"][5])
    ax.legend(loc = "upper center", bbox_to_anchor = (0.5, 1.3), ncol = 3)
    ax.set_xlabel("Y index (incl. guards)")
    ax.set_ylabel("X index (excl. guards)")


def plot_rz_grid(ds, ax, xlim = (None,None), ylim = (None,None)):
    
    m = ds.metadata
    
    rflat = ds.coords["R"].values.flatten()
    zflat = ds.coords["Z"].values.flatten()
    
    ax.set_title("R, Z space")
    ax.scatter(rflat, zflat, s = 0.1, c = "black", alpha = 0.5)
    ax.set_axisbelow(True)
    ax.grid()
    ax.plot(ds["R"][:,m["j1_1g"]], ds["Z"][:,m["j1_1g"]], label = "j1_1g",     color = m["colors"][0], alpha = 0.7)
    ax.plot(ds["R"][:,m["j1_2g"]], ds["Z"][:,m["j1_2g"]], label = "j1_2g", color = m["colors"][1], alpha = 0.7)
    ax.plot(ds["R"][:,m["j2_1g"]], ds["Z"][:,m["j2_1g"]], label = "j2_1g",     color = m["colors"][2], alpha = 0.7)
    ax.plot(ds["R"][:,m["j2_2g"]], ds["Z"][:,m["j2_2g"]], label = "j2_2g", color = m["colors"][3], alpha = 0.7)
    ax.plot(ds["R"][m["ixseps1"],:], ds["Z"][m["ixseps1"],:], label = "ixseps1", color = m["colors"][4], alpha = 0.7, lw = 2)

    ax.plot(
        ds["R"][m["ixseps1"], slice(None,m["ny_inner"])], 
        ds["Z"][m["ixseps1"], slice(None,m["ny_inner"])], 
        
        label = "ixseps1", color = m["colors"][4], alpha = 0.7, lw = 2)
    
    ax.plot(
        ds["R"][m["ixseps1"], slice(m["ny_inner"]+m["MYG"], m["nyg"])], 
        ds["Z"][m["ixseps1"], slice(m["ny_inner"]+m["MYG"], m["nyg"])], 
        
        color = m["colors"][4], alpha = 0.7, lw = 2)
    
    ax.plot(ds["R"][:,m["ny_inner"]], ds["Z"][:,m["ny_inner"]], label = "ny_inner", color = m["colors"][5], alpha = 0.7)
    
    ax.set_aspect("equal")
    # if ds.metadata["topology"] != "single-null":
    #     ax.plot(ds["R"][m["ixseps2"],:], ds["Z"][m["ixseps2"],:], label = "ixseps2", color = m["colors"][5], alpha = 0.7, lw = 2, ls=":")

    if xlim != (None,None):
        ax.set_xlim(xlim)
    if ylim != (None,None):
        ax.set_ylim(ylim)
        
def plot1d(
    casestore
    ):
    """
    Plot profiles of 1D hermes-3 cases
    provide dictionary of cases
    """
    lw = 2
    toplot = [ ["Te",  "Ne", "Pe"], 
            #   ["Ne", "Nd"]
            ]

    for list_params in toplot:

        fig, axes = plt.subplots(1,3, figsize = (18,5))
        fig.subplots_adjust(wspace=0.4)

        for i, ax in enumerate(axes):
            param = list_params[i]
            for i, casename in enumerate(casestore): 
                ds = casestore[casename].ds.isel(pos=slice(2,-2), t = -1)
                ax.plot(ds["pos"], abs(ds[param]), linewidth = lw, label = casename, marker = "o", ms = 0)

            ax.set_xlabel("Position (m)")
            ax.set_title(param)
            ax.legend(fontsize = 10)
            ax.grid(which="major", alpha = 0.3)

            if param in list_params:
                ax.set_ylim(0,ax.get_ylim()[1]*1.1)
                # ax.set_xlim(9,10.5)

            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1e}"))
def lineplot(
    cases,
    scale = "log",
    colors = ["teal", "darkorange", "deeppink", "limegreen", "firebrick",  "limegreen", "magenta","cyan", "navy"],
    params = ["Td+", "Te", "Td", "Ne", "Nd"],
    regions = ["imp", "omp", "outer_lower"],
    ylims = (None,None),
    xlims = (None,None),
    markersize = 2,
    dpi = 120,
    clean_guards = True
    ):
    
    """
    Provide a dictionary where key is name and value is a single time slice of a dataset
    """
    
    marker = "o"
    ms = markersize
    lw = 2.5
    set_ylims = dict()
    set_yscales = dict()
    
    # Automatically select last timestep if none provided
    for name in cases.keys():
        if "t" in  cases[name].dims.keys():
            cases[name] = cases[name].isel(t=-1)


    for region in regions:
        mult = 0.8
        fig, axes = plt.subplots(1,len(params), dpi = dpi, figsize = (4.6*len(params)*mult,5*mult), sharex = True)
        fig.subplots_adjust(hspace = 0, wspace = 0.3, bottom = 0.25, left = 0.1, right = 0.9)
        ls = "-"
        
        region_ds = dict()
        for name in cases.keys():
            ds = cases[name]
            if region == "omp":
                region_ds[name] = ds.hermesm.select_region("outer_midplane_a")
                xlabel = "Distance from separatrix [m]"
            elif region == "imp":
                region_ds[name] = ds.hermesm.select_region("inner_midplane_a")
                xlabel = "Distance from separatrix [m]"
            elif region == "outer_lower":
                region_ds[name] = ds.hermesm.select_region("outer_lower_target")
                xlabel = "Distance from separatrix [m]"
            elif region == "field_line":
                region_ds[name] = ds.hermesm.select_custom_sol_ring(ds.metadata["ixseps1"], "outer_lower").squeeze()
                xlabel = "Distance from midplane [m]"
            else:
                raise Exception(f"Region {region} not found")
            
            if clean_guards is True:
                if region == "omp" or region == "imp" or region == "outer_lower":
                    region_ds[name] = region_ds[name].isel(x=slice(2,-2))
                elif region == "field_line":
                    region_ds[name] = region_ds[name].isel(theta = slice(None,-2))
                else:
                    raise Exception(f"{region} guard cleaning not implemented")
                
            
        
            
        for i, param in enumerate(params):
            for j, name in enumerate(cases.keys()):
                
                
                
                if region == "field_line":    # Poloidal
                    m = region_ds[name].metadata
                    xplot = region_ds[name].coords["theta"] - region_ds[name].coords["theta"][0]
                else:    # Radial, 0 at sep
                    sep_R = region_ds[name].coords["R"][ds.metadata["ixseps1"]- ds.metadata["MXG"]]
                    xplot = region_ds[name].coords["R"] - sep_R
                    
                data = region_ds[name][param]
                
                
                    
                
                axes[i].plot(xplot, data, label = name, c = colors[j], marker = marker, ms = ms, lw = lw, ls = ls)
                
             
            if ylims != (None, None):
                axes[i].set_ylim(ylims)
            if xlims != (None, None):
                axes[i].set_xlim(xlims)
            
            axes[i].grid(which="both", alpha = 0.2)
            axes[i].set_xlabel(xlabel, fontsize=9)
            axes[i].set_yscale(scale)
            axes[i].set_title(f"{region}: {param}")
            axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=3,nbins=5))

            
        legend_items = []
        for j, name in enumerate(cases.keys()):
            legend_items.append(mpl.lines.Line2D([0], [0], color=colors[j], lw=2, ls = ls))
            
        fig.legend(legend_items, cases.keys(), ncol = len(cases), loc = "upper center", bbox_to_anchor=(0.5,0.15))
        # fig.tight_layout()
        
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


def camera_view(ax, loc, tokamak = "ST40"):
    """
    Available views:
    ---------
    lower_outer, lower2
    """
    
    lims = dict()
    if loc == "lower_outer":
        lims = dict(x = (0.45, 0.65), y = (-0.87, -0.7))
    elif loc == "lower2":
        lims = dict(x = (0.15, 0.66), y = (-0.87, -0.5))
    elif loc == "lower_half":
        lims = dict(x = (None,None), y = (-0.87, 0.1))
        
    else:
        raise Exception(f"Location {loc} not implemented yet")
        
    ax.set_xlim(lims["x"])
    ax.set_ylim(lims["y"])
    
    
def plot_perp_heat_fluxes(ds, ax = None, loc = "omp_integral", neutrals_only = False, ylim = (None,None)):
    """
    Plots poloidal integrals of radial heat fluxes
    Plot is for LHS cell edges and then the outermost RHS cell edge is appended
    This way zero radial flux appears as a "0" on both ends.
    """
    lw = 1
    
    if ds.coords["t"].shape != ():
        raise Exception("Must supply single time slice")
    
    if ax == None:
        fig, ax = plt.subplots()

    if loc == "integral":
        d = ds.isel(x=slice(2,-2)).sum("theta")
        print("Integrating poloidally")
        ylabel = "Radial heat flow $[MW]$"
        title = "Whole domain radial heat flow integral"
        scale = 1e-6
    elif loc == "omp":
        d = ds.hermesm.select_region("outer_midplane_a").isel(x=slice(2,-2))
        ylabel = "Radial heat flux $[Wm^{-2}]$"
        title = "Radial heat fluxes at OMP"
        scale = 1
    elif loc == "imp":    
        d = ds.hermesm.select_region("inner_midplane_a").isel(x=slice(2,-2))
        ylabel = "Radial heat flux $[Wm^{-2}]$"
        title = "Radial heat fluxes at IMP"
        scale = 1
    else:
        raise Exception("Location must be omp, imp or integral")

    
    # if "omp" in loc:
    #     d = domain.hermesm.select_region("outer_midplane_a").isel(x=slice(2,-2))
    # elif "imp" in loc:
    #     d = domain.hermesm.select_region("inner_midplane_a").isel(x=slice(2,-2))
    # else:
    #     raise Exception("Location must contain omp or imp")
        
    omp = ds.hermesm.select_region("outer_midplane_a").isel(x=slice(2,-2))
    dist = (omp["R"] - omp["R"][ds.metadata["ixseps1"]])
    dist = np.insert(dist.values, 0, dist.values[0] - (dist.values[1] - dist.values[0]))

    def append_rhs(x):
        F = d[x].values
        rhs = d[x.replace("_L_", "_R_")][-1].values
        return np.concatenate([F, [rhs]])

    m = "o"
    ms = 0
    
    if neutrals_only is False:
        ax.plot(dist, append_rhs("hf_perp_diff_L_e")*scale, label = "Electron total", marker = m, ms = 0, c = "teal")
        ax.plot(dist, append_rhs("hf_perp_diff_L_d+")*scale, label = "Ion conduction", marker = m, ms = ms, ls = "--", c = "red")
        ax.plot(dist, append_rhs("hf_perp_conv_L_d+")*scale, label = "Ion convection", marker = "x",ls = ":", ms = ms, c = "red")
        ax.plot(dist, append_rhs("hf_perp_tot_L_d+")*scale, label = "Ion total", marker = "|",ls = "-", ms = 0, c = "red")
    
    ax.plot(dist, append_rhs("hf_perp_diff_L_d")*scale, label = "Neutral conduction", marker = m, ls = "--", ms = ms, c = "dimgray")
    ax.plot(dist, append_rhs("hf_perp_conv_L_d")*scale, label = "Neutral convection", marker = "x",ls = ":", ms = ms, c = "dimgray")
    ax.plot(dist, append_rhs("hf_perp_tot_L_d")*scale, label = "Neutral total", marker = "|",ls = "-", ms = ms, c = "dimgray")
    
    leg = ax.legend(loc = "upper right", bbox_to_anchor = (1,1), fontsize = 10)
    
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    # domain["hf_perp_diff_R_d"].plot(ax = ax, label = "Neutral conduction")
    # domain["hf_perp_conv_R_d"].plot(ax = ax, label = "Neutral convection")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Distance from separatrix [m]")
    ax.set_title(title)
    ax.set_ylim(ylim)
    
def plot_perp_particle_fluxes(ds):
    """
    Plots poloidal integrals of radial particle fluxes
    Plot is for LHS cell edges and then the outermost RHS cell edge is appended
    This way zero radial flux appears as a "0" on both ends.
    """
    
    lw = 1
    
    def append_rhs(x):
        F = d[x]
        rhs = d[x.replace("_L_", "_R_")][-1].values
        return np.concatenate([F, [rhs]])
    
    if ds.coords["t"].shape != ():
        raise Exception("Must supply single time slice")
    
    fig, ax = plt.subplots(figsize=(4,3), dpi = 150)
    d = ds.isel(x=slice(2,-2)).sum("theta")
    
    Fi = append_rhs("pf_perp_diff_L_d+")
    Fn = append_rhs("pf_perp_diff_L_d")
    
    Ft = Fi + Fn
    ax.plot(range(len(Fi)), Fi,  marker = "o", label = f"d+", ms = 2, c = "teal", lw = lw)
    ax.plot(range(len(Fi)), Fn,  marker = "o", label = f"d", ms = 2, c = "darkorange", lw = lw)
    
    ax.set_xlabel("Radial index")
    ax.set_ylabel("Particle flow [s-1]")
    ax.set_title("Particle flow integral")
    ax.grid()
    ax.legend()
    
def plot_particle_balance(ds, ylims = (None, None)):
    """
    Plot net domain particle flows and the particle imbalance as a function of time.
    Particle flows are aggregated for all heavy species. 
    Requires you to have calculated the balance to begin with (see fluxes.py)
    """
    fig, ax = plt.subplots(figsize=(4,3), dpi = 150)
    m = ds.metadata
    data_pos = [ds["pf_int_core_net"], ds["pf_int_sol_net"], ds["pf_int_src_net"]]
    labels_pos = ["Core", "SOL", "Source"]

    data_neg = [ds["pf_int_targets_net"]]
    labels_neg = ["Targets"]

    # Ignore first X time for ylim calculation
    mins = []
    maxs = []
    for data in data_neg + data_pos:
        tlen = len(data.coords["t"])
        data = data.isel(t = slice(int(tlen*0.1), None))
        mins.append(np.nanmin(data.values))
        maxs.append(np.nanmax(data.values))
        
    max_magnitude = max( [abs(min(mins)), abs(max(maxs))] )
    ymin = - max_magnitude * 1.2
    ymax = max_magnitude * 1.2
    
    ax.stackplot(ds.coords["t"], data_pos, labels = labels_pos, baseline = "zero", colors = ["teal", "cyan", "navy"], alpha = 0.7)
    ax.stackplot(ds.coords["t"], data_neg, labels = labels_neg, baseline = "zero", colors = ["darkorange"], alpha = 0.7)

    ax.plot(ds.coords["t"], ds["pf_int_total_net"], lw = 2, ls = "--", c = "k", label = "Imbalance")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Particle flow [s-1]")
    ax.set_title("Particle balance")
    
    if ylims != (None, None):
        ax.set_ylim(ylims)
    else:
        ax.set_ylim(ymin,ymax)
    fig.legend(bbox_to_anchor = (1.25,0.9), loc = "upper right")
    ax.grid(lw = 0.5)
    
def plot_heat_balance(ds, ylims = (None, None)):
    """
    Plot net domain heat flows and the heat imbalance as a function of time.
    Heat flows are aggregated for all species. 
    Requires you to have calculated the balance to begin with (see fluxes.py)
    """
    fig, ax = plt.subplots(figsize=(4,3), dpi = 150)
    m = ds.metadata
    data_pos = [ds["hf_int_core_net"], ds["hf_int_sol_net"], ds["hf_int_src_net"]]
    labels_pos = ["Core", "SOL", "Source"]

    data_neg = [ds["hf_int_targets_net"]]
    labels_neg = ["Targets"]
    print(type(data_neg))
    if "hf_int_rad_ex_e" in ds.data_vars:
        data_neg.append(ds["hf_int_rad_ex_e"])
        labels_neg.append("Rad (ex)")
        print(type(data_neg))
    if "hf_int_rad_rec_e" in ds.data_vars:
        data_neg.append(ds["hf_int_rad_rec_e"])
        labels_neg.append("Rad (rec")

    # Ignore first X time for ylim calculation
    mins = []
    maxs = []
    for data in data_neg + data_pos:
        tlen = len(data.coords["t"])
        data = data.isel(t = slice(int(tlen*0.1), None))
        mins.append(np.nanmin(data.values))
        maxs.append(np.nanmax(data.values))
        
    max_magnitude = max( [abs(min(mins)), abs(max(maxs))] )
    ymin = - max_magnitude * 1.2
    ymax = max_magnitude * 1.2
    

    ax.stackplot(ds.coords["t"], data_pos, labels = labels_pos, baseline = "zero", colors = ["teal", "cyan", "navy"], alpha = 0.7)
    ax.stackplot(ds.coords["t"], data_neg, labels = labels_neg, baseline = "zero", colors = ["darkorange", "deeppink", "crimson"], alpha = 0.7)

    ax.plot(ds.coords["t"], ds["hf_int_total_net"], lw = 2, ls = "--", c = "k", label = "Imbalance")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Domain heat flow [W]")
    ax.set_title("Heat balance")
    
    
    if ylims != (None, None):
        ax.set_ylim(ylims)
    else:
        ax.set_ylim(ymin,ymax)
    fig.legend(bbox_to_anchor = (1.25,0.9), loc = "upper right")
    ax.grid(lw = 0.5)
    
def plot_omp(da_list, legend = True, title = True, dpi = 150, **kwargs):
    """
    Wrap standard Xarray plotter for the outboard midplane
    """
    fig, ax = plt.subplots(figsize=(4,3), dpi = dpi)
    for da in da_list:
        da.hermesm.select_region("outer_midplane_a").plot(ax = ax, label = da.standard_name, **kwargs)
        
    ax.grid()
    if legend is True and len(da_list)>1:
        ax.legend()
    
    if len(da_list) == 1 and title is True:
        ax.set_title(da_list[0].standard_name)
    else:
        ax.set_title("")
    

def plot_density_feedback_controller(ds):
    """
    Plot the upstream density, and the feedback signal breakdown
    """
    fig, axes = plt.subplots(1,3, figsize=(4.5*3, 3.5), dpi = 120)
    fig.subplots_adjust(wspace=0.3, bottom = 0.2)
    colors = ["teal", "darkorange", "firebrick", "deeppink", "green", "navy"]

        
    ds = ds.isel(pos=slice(2,-1))
    particle_count = ((ds["Ne"] + ds["Nd"])*ds["dv"]).sum("pos")
    ds["Ne"].isel(pos=0).plot(ax = axes[0], c = colors[0])
    # ds["densi"]
    
    ds["density_feedback_src_p_d+"].plot(ax = axes[1], c = "darkorange", label = "P")
    ds["density_feedback_src_i_d+"].plot(ax = axes[1], c = "purple", label = "I")
    
    ds["density_feedback_src_mult_d+"].plot(ax = axes[1], c = colors[0], ls = ":", label = "Total")


    # pflux.plot(ax=axes[2])
    axes[0].set_title("Upstream density")
    axes[1].set_title("Signal breakdown")
    axes[1].legend()
    axes[2].set_title("Relative error")
    axes[1].set_ylabel("-")


    for ax in axes:
        ax.grid()