#!/usr/bin/env python3

from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import traceback
import platform
import colorcet as cc
from scipy import stats
from boututils.datafile import DataFile
from boutdata.collect import collect
from boutdata.data import BoutData
import xbout


class Case:
    """SD1D case"""

    def __init__(self, casepath, verbose = False, load = True):
        self.casename = casepath.split(os.path.sep)[-1]
        self.casepath = casepath
        self.datapath = os.path.join(casepath, "BOUT.dmp.*.nc")
        self.inputfilepath = os.path.join(casepath, "BOUT.inp")
        self.verbose = verbose

        if load:
            self.load_data()
            self.extract_variables()
            self.process_variables()

    def load_data(self):
        self.boutdata = BoutData(self.casepath, yguards=True, strict = True)
        self.raw_data = self.boutdata["outputs"]
        self.options = self.boutdata["options"]

        self.get_timestats()

    def extract_variables(self, tind = -1):
        self.missing_vars = []
        self.norm_data = dict()

        var_collect = ["P", "Ne", "Nn", "NVi", "NVn", "kappa_epar", "Pn", "Dn",
                        "S", "Srec", "Siz", # Plasma particles and sinks
                        "F", "Frec", "Fiz", "Fcx", "Fel", # Momentum source/sinks to neutrals
                        "R", "Rrec", "Riz", "Rzrad", "Rex", # Radiation, energy loss from system
                        "E", "Erec", "Eiz", "Ecx", "Eel", # Energy transfer between neutrals and plasma
                        "Pe", "Pd+", "Ve", "Vd+", "Nd", "Nd+", "Td+", "SPd+", "SNd+", "Ert",
                        "PeSource", "NeSource"]

        #------------Unpack data
        for var in var_collect:
            try:
                data = self.raw_data[var].squeeze()
                if len(data.shape) > 1:
                    data = data[tind,1:-1]
                else:
                    data = data[1:-1]
                self.norm_data[var] = data

            except Exception:
                # traceback.print_exc()
                self.missing_vars.append(var)
        
        # Normalisation constants
        for var in ["Nnorm", "Tnorm", "Cs0", "Omega_ci", "rho_s0"]:
            setattr(self, var, self.raw_data[var])

        # Geometry data
        for var in ["dy", "J", "g_22"]:
            setattr(self, var, self.raw_data[var][0, 1:-1])

        #------------Identify solved physics
        if "Vd+" in self.missing_vars:
            self.hermes = False
        else:
            self.hermes = True

        if "Pe" not in self.missing_vars:
            self.ion_eqn = True
        else:
            self.ion_eqn = False

        if "Pn" in self.options.keys():
            if self.options["Pn"]["evolve"] == "true":
                self.evolve_pn = True
            else:
                self.evolve_pn = False
        else:
            self.evolve_pn = False

        if "NVn" in self.options.keys():
            if self.options["NVn"]["evolve"] == "true":
                self.evolve_nvn = True
            else:
                self.evolve_nvn = False
        else:
            self.evolve_nvn = False

        #------------Derive variables
        self.dV = self.dy * self.J
        
        if self.hermes:
            self.norm_data["Nn"] = self.norm_data["Nd"]
            self.norm_data["Vi"] = self.norm_data["Vd+"]
            self.norm_data["NVi"] = self.norm_data["Vd+"] * self.norm_data["Nd+"]
            self.norm_data["P"] = self.norm_data["Pe"] + self.norm_data["Pd+"]
            self.norm_data["S"] = self.norm_data["SNd+"] * -1
            self.norm_data["Ti"] = self.norm_data["Td+"]

        self.norm_data["Vi"] = self.norm_data["NVi"] / self.norm_data["Ne"]

        if self.ion_eqn:
            self.norm_data["Te"] = (self.norm_data["Pe"] / self.norm_data["Ne"] ) # Electron temp
        else:
            self.norm_data["Te"] = (0.5 * self.norm_data["P"] / self.norm_data["Ne"] ) # Electron temp

        if self.verbose:
            print(self.missing_vars)

    def process_variables(self):

        norm = self.norm_data
        dnorm = dict()

        #------------Denormalisation
        """
        NORMALISATION FACTORS AND UNITS
        Nnorm : Density : [m-3]
        Tnorm : Temperature : [eV]
        Pnorm : Pressure : [Pa]
        Snorm : Density flux : [m-3s-1]
        Fnorm : Momentum : [kgm-2s-2, or Nm-3]
        Enorm : Power : [Wm-3]
        Dnorm : Diffusion [m2s-1]
        Cs0 : Speed : [ms-1]
        Omega_ci : Time : [s-1]
        rho_s0 : Length : [m]
        """

        # Derived normalisation factors
        self.Pnorm = self.Nnorm * self.Tnorm * constants("q_e") # Converts P to Pascals. 1.602e-19 is proton charge in C
        self.Snorm = self.Nnorm * self.Omega_ci # Normalisation for S: plasma density sink (m-3s-1)
        self.Fnorm = (constants("mass_p") * 2) * self.Nnorm * self.Cs0 * self.Omega_ci # Plasma momentum sink normalisation (kgm-2s-1)
        self.Enorm = constants("q_e") * self.Nnorm * self.Tnorm * self.Omega_ci # Plasma energy sink normalisation
        self.Dnorm = self.rho_s0 * self.rho_s0 * self.Omega_ci # [m][m][s-1]=[m^2/s] diffusion coefficient
        
        list_tnorm = ["Te", "Td+"] # [eV]
        list_nnorm = ["Ne", "Nn", "Nd+", "Nd"] # [m-3]
        list_pnorm = ["P", "Pn", "dynamic_p", "dynamic_n", "Pe", "SPd+", "Pd+"] # [Pa]
        list_snorm = ["S", "Srec", "Siz", "NeSource", "SNd+"] # [m-3s-1]
        list_fnorm = ["F", "Frec", "Fiz", "Fcx", "Fel"] # [kgm-2s-2 or Nm-3]
        list_enorm = ["E", "R", "Rrec", "Riz", "Rzrad", "Rex", "Erec", "Eiz", "Ecx", "Eel", "Ert", "PeSource"] # Wm-3
        list_dnorm = ["Dn"]
        list_vnorm = ["Vi", "Ve", "Vd+"]
        list_fluxnorm = ["NVi", "NVn"]

        for var in norm:
            if var in list_tnorm:
                dnorm[var] = norm[var] * self.Tnorm
            elif var in list_nnorm:
                dnorm[var] = norm[var] * self.Nnorm
            elif var in list_pnorm:
                dnorm[var] = norm[var] * self.Pnorm
            elif var in list_snorm:
                dnorm[var] = norm[var] * self.Snorm
            elif var in list_fnorm:
                dnorm[var] = norm[var] * self.Fnorm
            elif var in list_enorm:
                dnorm[var] = norm[var] * self.Enorm
            elif var in list_dnorm:
                dnorm[var] = norm[var] * self.Dnorm
            elif var in list_vnorm:
                dnorm[var] = norm[var] * self.Cs0
            elif var in list_fluxnorm:
                dnorm[var] = norm[var] * self.Nnorm * self.Cs0

        # Sheath stats
        self.flux_out = (norm["NVi"][-1] * self.J / np.sqrt(self.g_22) * self.Omega_ci * self.Nnorm)[0] # Sheath flux in [s-1]. J/g_22 is cross-sectional area (J*dy)/(sqrt(g_22)*dy) = Volume/length

        # Reconstruct grid position from dy
        n = len(self.dy)
        self.pos = np.zeros(n)
        self.pos[0] = -0.5*self.dy[1]
        self.pos[1] = 0.5*self.dy[1]

        for i in range(2,n):
            self.pos[i] = self.pos[i-1] + 0.5*self.dy[i-1] + 0.5*self.dy[i]

        
        #------------Guard replacement, geometry and BC handling
        self.pos = replace_guards(self.pos)

        for var in dnorm.keys():
            if var in ["Vi", "P", "Ne", "Nn", "kappa_epar", "Pn", "Dn", "Te"]: # NVi already has its last value as the sheath flux.
                if var in dnorm.keys():
                    dnorm[var] = replace_guards(dnorm[var])
                
            # There are no sources in guard cells, so let's set them to NaN so they can be plotted
            # against a grid that includes the guard cells.
            elif var not in ["NVi", "NVn"]:
                dnorm[var][-1] = np.nan
                dnorm[var][0] = np.nan
                    
        # Correct Te to reflect BC condition within code
        dnorm["Te"][-1] = dnorm["Te"][-2]

        #------------Derived variables
        dnorm["Ntot"] = dnorm["Ne"] + dnorm["Nn"]
        dnorm["Vi"] = dnorm["NVi"] / dnorm["Ne"]
        dnorm["Cs"] = self.Cs0 * np.sqrt(2 * dnorm["Te"]/self.Tnorm) # Sound speed
        dnorm["M"] = dnorm["Vi"] / dnorm["Cs"]
        dnorm["dynamic_p"] = norm["NVi"]**2 / norm["Ne"] * self.Pnorm
        
        dnorm["Ne_avg"] = sum(dnorm["Ne"] * self.dV)/sum(self.dy)
        dnorm["Nn_avg"] = sum(dnorm["Nn"] * self.dV)/sum(self.dy)
        dnorm["Ntot_avg"] = dnorm["Ne_avg"] + dnorm["Nn_avg"]

        if self.hermes:
            dnorm["SEd+"] = dnorm["SPd+"] * 3/2

        else:
            dnorm["ESource"] = dnorm["PeSource"] * 3/2 # Is this right?

        if self.evolve_nvn:
            dnorm["dynamic_n"] = norm["NVn"]**2 / norm["Nn"] * self.Pnorm

        #------------Add geometry terms for completion
        dnorm["pos"] = self.pos
        dnorm["dy"] = self.dy
        dnorm["J"] = self.J
        dnorm["dV"] = self.dV

        self.data = dnorm

    def plot_residuals(self, params = ["Te", "Ne", "NVi", "S", "R"], \
                   list_plot = ["Te", "Ne", "NVi", "S", "R"], \
                   skip = 1, smoothing = 1, norm_range = 10, normalise = True):

        """
        plot_residuals plots residuals of selected variables
        Residuals here are defined as sum of absolute changes in a 
        variable over all cells, absolute.
        This is plotted on a log chart with smoothing settings.

        Settings:
        case: case name
        params: variables to collect and calculate
        skip: 1 means no skipping, 2 means take every second point, etc.
        smoothing: moving average period for nice plotting
        norm_range: number of initial datapoints to normalise rest by
        """
        
        num_timesteps = len(collect(
            "Ne", path = self.casepath, yguards=True, info = False, strict = True)[:,0,0,0]
                        )
        
        
        BoutData(self.casepath)["options"]["NOUT"]
        resolution = int(num_timesteps/skip) #skip of 2 means half the points

        # Calculate steps. Sequence starts after first timestep
        # and ends at the final number, which is num_timesteps - 1.
        # This is achieved by manually subtracting 1 from final value.
        stepsize = int(num_timesteps/resolution) 
        steps = np.array(range(stepsize, num_timesteps + stepsize, stepsize))
        steps[-1] = steps[-1] - 1

        # Hist is array of parameter over all timesteps.
        # Norm is normalisation factor.

        hist = dict()
        norm = dict()

        # df_res holds the residuals. Norm holds the normalised ones.
        # Not all residuals are stored, only the ones every stepsize.
        df_res = pd.DataFrame()
        df_norm = df_res.copy()
        df_rolling = pd.DataFrame()

        for param in params:
            
            try:
                hist = collect(param, path = self.casepath, yguards=True, info = False, strict = True)[:,0,1:-1,0]
            except:
                print("Could not collect {}".format(param))
                continue
            # Calculate the residual. It is defined by the difference of the sums of the absolute
            # values of a parameter between the current and last timestep. This effectively
            # means "how much has this parameter changed in all the cells". It is absolute.
            res = np.zeros(num_timesteps)
            for t in range(1, num_timesteps):
                res[t] =  ( abs(  sum(abs(hist[t,:])) - sum(abs(hist[t-1,:]))  ))

            # Here we make a reduced array where we only sample every step. 
            # This is because otherwise the arrays get large and plots get too busy.
            reduced = np.zeros(resolution)
            for i, step in enumerate(steps):
                reduced[i] = res[step]

            # Applying normalisation, moving average and filling dataframes.
            norm[param] = max((res[:norm_range]))
            df_res[param] = reduced
            df_norm[param] = reduced / norm[param]
            
            if normalise == True:
                df_rolling[param] = df_norm[param].rolling(smoothing).mean()
            else:
                df_rolling[param] = df_res[param].rolling(smoothing).mean()


        print("Complete")

        # Plot
        fig, ax = plt.subplots(figsize = (11,6))

        for param in list_plot:
            try:
                ax.plot(steps, df_rolling[param], label = param, linewidth = 2)
            except:
                pass

        # Legend outside of plot
        leg = fig.legend(bbox_to_anchor=(0.95, 0.5), loc = "center", fontsize = 12)

        # Thick lines in legend
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3)

        ax.grid(which = "major", alpha = 1)
        ax.grid(which = "minor", alpha = 0.2)
        ax.set_ylabel("Normalised residuals", fontsize = 12)
        ax.set_xlabel("Timestep", fontsize = 12)
        ax.set_title(f"{self.casename} || Normalised residual plot || smoothing: {smoothing}", fontsize = 12) 
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1)
        ax.tick_params(axis = "both", which = "major", labelsize = 12)
        ax.tick_params(axis = "both", which = "minor", labelsize = 12)

    def mass_balance(self,          
        verbosity = True, 
        plot = False, 
        output = False, 
        nloss_guardcells = True, 
        flux_input = False,
        NVi_is_NeVi = False,
        additional_plots = False,
        error_only = True):
        
        """ 
        path_case, << Case path          
        verbosity = False, << Print output
        plot = True,  << Pretty pictures
        output = False, << Output key metrics as a dict
        nloss_guardcells = True,  << count guardcells for nloss calculation (gives right answer if True)
        flux_input = False,  << if using constant specified flux in the input file
        NVi_is_NeVi = False  << Calculate NVi as product of Ne and Vi (makes no difference)
        additional_plots = False  << Extra plots include history of mass in case.
        error_only = True  << set to False to get a full mass balance plot.
        """
        
        path_case = self.casepath

        # d for time independent data, h for time histories
        d = dict()
        h = dict()

        # Data collection
        d["J"] = collect("J", path = path_case, yguards = True, info = False)[0,1:-1]
        d["g_22"] = collect("g_22", path = path_case, yguards = True, info = False)[0,1:-1]
        h["Ne"] = collect("Ne", path = path_case, yguards = True, info = False, strict = True)[:,0,1:-1,0]
        h["Vi"] = collect("Vi", path = path_case, yguards = True, info = False, strict = True)[:,0,1:-1,0]
        h["NVi_original"] = collect("NVi", path = path_case, yguards = True, info = False, strict = True)[:,0,1:-1,0]
        d["Nnorm"] = collect("Nnorm", path = path_case)
        d["Cs0"] = collect("Cs0", path = path_case)
        d["Omega_ci"] = collect("Omega_ci", path = path_case)
        d["dy"] = collect("dy", path = path_case, yguards = True, info = False, strict = True)[0,1:-1]
        d["dy_full"] = collect("dy", path = path_case, yguards = True, info = False, strict = True)[0,:] 
        d["J_full"] = collect("J", path = path_case, yguards = True, info = False, strict = True)[0,:] 
        nloss = self.options["sd1d"]["nloss"]
        frecycle_intended = self.options["sd1d"]["frecycle"]
        Nu_target = self.options["sd1d"]["density_upstream"]

        if flux_input == True:
            flux_intended = BoutData(path_case)["options"]["Ne"]["flux"]


        # We don't always have atomics enabled, but if we do, collect plasma density sink
        # Note if we did... this is the "sink" flag
        try:
            h["S"] = collect("S", path = path_case, yguards = True, info = False, strict = True)[:,0,1:-1,0]
            sink = True
        except:
            sink = False
            pass


        # We don't always have atomics present, but if we do, collect neutral density and do ze flag.
        try:
            h["Nn"] = collect("Nn", path = path_case, yguards = True, info = False, strict = True)[:,0,1:-1,0]
            h["Nn_full"] = collect("Nn", path = path_case, yguards = True, info = False, strict = True)[:,0,:,0]
            neutrals = True
        except:
            neutrals = False
            pass


        # NeSource is different because sometimes it's set as constant (like in case-01)
        # This means it has no length in the time dimension and things break.
        # Note whether there is a time dimension or not and store this as "evolving_source" flag
        NeSource = collect("NeSource", path = path_case, yguards = True, info = False, strict = True)
        if len(NeSource.shape) == 3: 
            h["NeSource"] = NeSource[:,0,1:-1]
            evolving_source = True
        elif len(NeSource.shape) == 2:
            d["NeSource"] = NeSource[0,1:-1]
            evolving_source = False

        del NeSource

        num_points = len(d["J"])
        num_timesteps = len(h["Ne"][:,0])


        #>>>>>>> NVi calculation                           
        for t in range(num_timesteps):
            h["Ne"][t,:] = replace_guards(h["Ne"][t,:])     
            h["Vi"][t,:] = replace_guards(h["Vi"][t,:]) 
            if neutrals:
                h["Nn"][t,:] = replace_guards(h["Nn"][t,:])

        h["NVi"] = h["Ne"] * h["Vi"] # NVi calculated like this because the SD1D NVi variable is already guard replaced and would cause confusion
        h["NVi_dn"] = h["NVi"] * d["J"]/np.sqrt(d["g_22"]) * d["Omega_ci"] * d["Nnorm"]
        h["NVi_original_dn"] = h["NVi_original"] * d["J"]/np.sqrt(d["g_22"]) * d["Omega_ci"] * d["Nnorm"]

        # This part is a bit hacky but it turns out that doing Ne*Vi is precisely the same as NVi anyway.
        if NVi_is_NeVi == True:
            d["NVi_out_dn"] = h["NVi_dn"][:,-1]   
        else:
            # NVi is NVi from the code. Not guard replaced.
            d["NVi_out_dn"] = h["NVi_original_dn"][:,-1]

        d["Nu_dn"] = h["Ne"][:,0] * d["Nnorm"]


        #>>>>>>> NeSource calculation    
        # Guard replacement gives wrong answer
        # First guard cell has non-zero value while
        # it should be zero. We should only be counting the
        # values in the domain, not in the guards, and not guard replacing
        # This is because the sources are already per volume, so they don't
        # depend on cell size - and so any guard replacement change makes them wrong.
        if evolving_source == True:
            h["NeSource"][:,0] = 0
            h["NeSource"][:,-1] = 0
            h["NeSource_dn"] = h["NeSource"] * d["dy"] * d["J"] * d["Omega_ci"] * d["Nnorm"] 
            d["NeSource_int_dn"] = np.sum(h["NeSource_dn"], axis = 1)

        if evolving_source == False:
            d["NeSource"][0] = 0
            d["NeSource_dn"] = d["NeSource"] * d["dy"] * d["J"] * d["Omega_ci"] * d["Nnorm"]
            d["NeSource_int_dn"] = np.ones(num_timesteps) * np.sum(d["NeSource_dn"])


        #>>>>>>> S calculation 
        # Treat it the same as NeSource. Only count the domain, not the 
        # guard cells, and do not guard replace.
        if sink == True:
            h["S"][:,0] = 0
            h["S"][:,-1] = 0
            h["S_dn"] = h["S"] * d["dy"] * d["J"] * d["Omega_ci"] * d["Nnorm"]
            d["S_int_dn"] = np.sum(h["S_dn"], axis = 1)


        #>>>>>>> nloss calculation 
        if neutrals:
            if nloss_guardcells:
            # Count nloss in all guard cells.
                h["nloss_dn"] = nloss * h["Nn_full"] * d["Nnorm"] * d["dy_full"] * d["J_full"] 
            else:
            # Do not count nloss in any guard cells.
                h["nloss_dn"] = nloss * h["Nn"] * d["Nnorm"] * d["dy"] * d["J"] 

            d["nloss_int_dn"] = np.sum(h["nloss_dn"], axis = 1)
            flux_nloss = d["nloss_int_dn"] #/ d["Omega_ci"]
        else:
            flux_nloss = np.zeros(num_timesteps)


        #>>>>>>> Final flux balance calculations
        flux_source = d["NeSource_int_dn"] # Source integral at last timestep
        flux_sheath = d["NVi_out_dn"] # Sheath flux at last timestep

        if sink == True:
            flux_sink = d["S_int_dn"] # Atomic sink integral at last timestep
            flux_in = flux_source - flux_sink
            if neutrals:
                frecycle = abs((flux_sink - flux_nloss)/flux_sheath)
            else:
                frecycle = abs((flux_sink)/flux_sheath)
            frecycle_error = abs(frecycle_intended - frecycle)
        else:
            flux_sink = np.array([0] * len(flux_source))
            frecycle = np.array([0] * len(flux_source))
            frecycle_error = np.array([0] * len(flux_source))
            flux_in = flux_source

        flux_imbalance = (flux_in - flux_sheath)/flux_sheath


        #>>>>>>> Mass conservation calculations
        # Is the total plasma + neutral loss going up or down?
        # Find average rate of mass loss or gain
        d["Ne_int"] = np.sum(h["Ne"] * d["J"] * d["dy"] * d["Nnorm"], axis = 1)
        if neutrals:
            d["Nn_int"] = np.sum(h["Nn"] * d["J"] * d["dy"] * d["Nnorm"], axis = 1)   
            d["N_int"] = d["Ne_int"] + d["Nn_int"]
        else:
            d["N_int"] = d["Ne_int"]

        # Ignore start of solution
        trim = int(np.floor(num_timesteps/4))
        d["N_int_trim"] = d["N_int"][trim:]
        d["idx_trim"] = np.array(range(trim,len(d["N_int"])))

        # Smooth data
        ma_period = int(np.floor(num_timesteps/2))
        d["N_int_ma"] = np.convolve(d["N_int_trim"], np.ones(ma_period)/ma_period, mode = "valid") 
        d["ma_xpoints"] = np.array(range(len(d["N_int_ma"])))+trim

        # Regression
        d["N_ma_fit"] = stats.linregress(d["ma_xpoints"], d["N_int_ma"])
        d["N_ma_fit"].slope
        d["N_ma_regr"] = d["ma_xpoints"] * d["N_ma_fit"].slope + d["N_ma_fit"].intercept
        d["mass_gain_rate"] = d["N_ma_fit"].slope
        d["mass_rate_frac"] = d["mass_gain_rate"] / np.mean(d["N_int_ma"])

        #>>>>>>> Upstream density error calculation
        if Nu_target > 0:
            Nu_error =  (d["Nu_dn"] - Nu_target) / Nu_target
        else:
            Nu_error = np.zeros(len(d["Nu_dn"]))


        if verbosity == True:
            print("\n>>Mass balance V5")
            #>>>>>>> Print output

            print(">>Case: {}".format(path_case))
            print(f"- Evolving particle source: {evolving_source}")
            print(f"- Atomic plasma sink: {sink}")
            print("--------------------------------")
            print(">>Summary of final timestep fluxes [s-1]:")
            print(f"- Final NeSource integral: {flux_source[-1]:.3E}")
            print(f"- Final NVi flux: {flux_sheath[-1]:.3E}")
            if flux_input:
                print(f"- Input file flux: {flux_intended:.3E}")
            if sink:
                print(f"- Final density sink integral: {flux_sink[-1]:.3E}")
            if neutrals:
                print(f"- Input file neutrals loss: {flux_nloss[-1]:.3E}")
            print("--------------------------------")
            print(f"- Final total particles in: {flux_in[-1]:.3E}")
            print(f"- Final total particles out: {flux_sheath[-1]:.3E}")
            print("- Average particle change per timestep: {:.3E}".format(d["mass_gain_rate"]))
            print("--------------------------------")
            print(f"- Intended recycle fraction: {frecycle_intended:.1%}")
            print(f"- Actual recycle fraction: {frecycle[-1]:.1%}")
            print("--------------------------------")
            print(f">>Final mass imbalance: {flux_imbalance[-1]:.2%}")
            print(f">>Final recycle error: {frecycle_error[-1]:.2%}")
            print(">>Mass change rate as fraction of total particles: {:.3E}".format(d["mass_rate_frac"]))
            print("--------------------------------")

        if plot == True:
            
            
            if error_only == True:
                
                ### MASS BALANCE HISTORY PLOT
                fig, ax = plt.subplots(figsize = (6,6))

                ax.set_yscale("linear")

                ax.plot(Nu_error, color = "green", linewidth = 2, 
                        label = f"Nu error, final={Nu_error[-1]:.2%}", linestyle = "solid")
                ax.plot(flux_imbalance, color = "red", linewidth = 2, 
                            label = f"Mass error, final={flux_imbalance[-1]:.2%}", linestyle = "solid")
                ax.plot(frecycle_error, color = "red", linewidth = 2, 
                            label = f"Recycle error, final={frecycle_error[-1]:.2%}", linestyle = "dashed")

                ax.set_ylim(-0.3,0.3)
                ax.yaxis.set_major_formatter("{x:.0%}")
                ax.set_ylabel("Error", fontsize = 12)

                ax.grid(which="major", color = "black", linestyle = "dotted")
                ax.grid(which="minor", alpha = 1)
                ax.set_xlabel("Timestep", fontsize = 12)
                ax.set_title("Control errors for {}".format(self.casename), fontsize = 10)
                ax.tick_params(axis = "both", which = "major", labelsize = 12)
                ax.tick_params(axis = "both", which = "minor", labelsize = 12)

                # leg = ax.legend(fontsize = 12)
                fig.legend(loc="upper right", bbox_to_anchor=(1.75,1), bbox_transform=ax.transAxes, fontsize = 12)

            else:
                        ### MASS BALANCE HISTORY PLOT
                fig, ax = plt.subplots(figsize = (6,6))
                ax.plot(d["NVi_out_dn"]+(d["nloss_int_dn"]), color = "black", linewidth = 2, label = f"Sheath+nloss sink", linestyle = "solid")
                ax.stackplot(range(len(d["NVi_out_dn"])), d["S_int_dn"]*-1, d["NeSource_int_dn"], labels = ["Ionisation source", "Flux source"], alpha = 0.5)

                ax.set_yscale("linear")
                ax2 = ax.twinx()
                ax2.plot(Nu_error, color = "green", linewidth = 1, 
                        label = f"Nu error, final={Nu_error[-1]:.2%}", linestyle = "solid")
                ax2.plot(flux_imbalance, color = "red", linewidth = 1, 
                            label = f"Mass error, final={flux_imbalance[-1]:.2%}", linestyle = "solid")
                ax2.plot(frecycle_error, color = "red", linewidth = 1, 
                            label = f"Recycle error, final={frecycle_error[-1]:.2%}", linestyle = "dashed")


                ax2.set_ylim(-0.3,0.3)
                ax2.yaxis.set_major_formatter("{x:.0%}")
                ax2.spines["right"].set_color("red")
                ax2.yaxis.label.set_color("red")
                ax2.title.set_color("red")
                ax2.set_ylabel("Error", fontsize = 12)
                ax2.tick_params(axis="y", colors="red")

                ax.grid(which="major", color = "black", linestyle = "dotted")
                ax.grid(which="minor", alpha = 1)
                ax.set_xlabel("Timestep", fontsize = 12)
                ax.set_ylabel("Flux", fontsize = 12)
                ax.set_title("Mass balance for {}".format(self.casename), fontsize = 10)
                ax.tick_params(axis = "both", which = "major", labelsize = 12)
                ax.tick_params(axis = "both", which = "minor", labelsize = 12)

                # leg = ax.legend(fontsize = 12)
                fig.legend(loc="upper right", bbox_to_anchor=(1.75,1), bbox_transform=ax.transAxes, fontsize = 12)            
            if additional_plots == True:

                    ### MASS BALANCE AND RECYCLE ERROR HISTORY
                    fig, ax = plt.subplots(figsize = (6,4))
                    ax.plot(flux_imbalance, color = "black", linewidth = 1, 
                            label = f"Mass imbalance {flux_imbalance[-1]:.2%}")

                    if sink == True:
                        ax.plot(frecycle_error, color = "red", linewidth = 1, 
                                label = f"Recycle error {frecycle_error[-1]:.2%}")
                    ax.set_yscale("log")
                    leg = ax.legend(fontsize = 12)
                    ax.grid(which="major")
                    ax.grid(which="minor", alpha = 0.2)
                    ax.set_xlabel("Timestep", fontsize = 12)
                    ax.set_ylabel("Error fraction", fontsize = 12)
                    ax.set_title("Mass balance error for {}".format(self.casename), fontsize = 10)
                    ax.tick_params(axis = "both", which = "major", labelsize = 12)
                    ax.tick_params(axis = "both", which = "minor", labelsize = 12)

                    for legobj in leg.legendHandles:
                        legobj.set_linewidth(3)

                    ### TOTAL MASS HISTORY
                    fig, ax = plt.subplots(figsize = (6,4))
                    ax.plot(d["ma_xpoints"], d["N_int_ma"], color = "black", linewidth = 1)
                    ax.plot(d["ma_xpoints"], d["N_ma_regr"], color = "red", linestyle = "--", 
                            linewidth = 2, label = "Fractional change \nevery tstep: {:.2E}".format(d["mass_rate_frac"]))


                    ax.set_yscale("log")
                    leg = ax.legend(fontsize = 12)
                    ax.grid(which="major")
                    ax.grid(which="minor", alpha = 0.2)
                    ax.set_xlabel("Timestep: MA period {}".format(ma_period), fontsize = 12)
                    ax.set_ylabel("Particles", fontsize = 12)
                    ax.set_title("Total mass history for {}".format(self.casename), fontsize = 10)
                    ax.tick_params(axis = "both", which = "major", labelsize = 12)
                    ax.tick_params(axis = "both", which = "minor", labelsize = 12)
                    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.3E}"))
            
        if output == True:
            return {"NeSource" : flux_source[-1], "S" : flux_sink[-1], 
                    "NVi" : flux_sheath[-1], "input_flux" : flux_intended,
                    "nloss" : flux_nloss[-1], "input_frecycle": frecycle_intended, 
                    "actual_frecycle" : frecycle[-1], "recycle_error" : frecycle_error[-1],
                    "imbalance" : flux_imbalance[-1], "mass_rate_frac" : d["mass_rate_frac"]
                    }

    def animate(self, param):
        ds = xbout.open_boutdataset(datapath = self.datapath, inputfilepath = self.inputfilepath,
                                    info = False, keep_yboundaries = True)
        ds = ds.squeeze(drop = True)
        xbout.plotting.animate.animate_line(ds[param])

    def get_timestats(self):

            self.wtime = self.raw_data["wtime"]
            self.wtime_sum = sum(self.raw_data["wtime"])/3600
            self.wtime_avg = np.mean(self.wtime)
            self.wtime_std = np.std(self.wtime)
            self.rhscalls = self.raw_data["wtime_rhs"]
            self.rhscalls_sum = sum(self.raw_data["wtime_rhs"])
            

class CaseDeck:
    def __init__(self, path, key = "", keys = [], verbose = False):

        self.casepaths_all = dict()
        self.casepaths = dict()
        self.cases = dict()
        for root, dirs, files in os.walk(path):
            for file in files:
                if ".dmp" in file:
                    case = os.path.split(root)[1]
                    self.casepaths_all[case] = root

                    if key != "":
                        if key in root:
                            self.casepaths[case] = root

                    elif keys == []:
                        self.casepaths[case] = root

                    if keys != []:
                        if any(x in case for x in keys):
                            self.casepaths[case] = root

        self.casenames_all = list(self.casepaths_all.keys())
        self.casenames = list(self.casepaths.keys())

        if verbose:
            print("\n>>> All cases in path:", self.casenames_all)
            print(f"\n>>> All cases matching the key '{key}': {self.casenames}\n")
        
        print(f">>> Loading cases: ", end="")

        for case in self.casenames:
            self.cases[case] = Case(self.casepaths[case])
            print(f"{case}... ", end="")

        self.get_stats()
        
        print("...Done")


    def get_stats(self):
        self.stats = pd.DataFrame()
        self.stats.index.rename("case", inplace = True)

        for casename in self.casenames:
            case = self.cases[casename]
            if case.hermes:
                Nnorm = case.options["hermes"]["Nnorm"]
                self.stats.loc[casename, "initial_dens"] = case.options["Nd+"]["function"] * Nnorm
                self.stats.loc[casename, "line_dens"] = case.options["Nd+"]["function"] * Nnorm
                self.stats.loc[casename, "target_flux"] = case.data["NVi"][-1]
                self.stats.loc[casename, "target_temp"] = case.data["Te"][-1]

        self.stats.sort_values(by="initial_dens", inplace=True)


    def plot(self, vars = [["Te", "Ne", "Nn"], ["S", "R", "P"], ["NVi", "M", "F"]]):
        lib = library()


        colors = mike_cmap(10)
        lw = 2

        for list_params in vars:

            fig, axes = plt.subplots(1,3, figsize = (18,5))
            fig.subplots_adjust(wspace=0.4)

            for i, ax in enumerate(axes):
                param = list_params[i]
                for i, case in enumerate(self.casenames):
                    data = self.cases[case].data
                    ax.plot(data["pos"], data[param], color = colors[i], linewidth = lw, label = case)


                ax.set_xlabel("Position (m)")
                ax.set_ylabel("{} ({})".format(lib[param]["name"], lib[param]["unit"]), fontsize = 11)
                ax.set_yscale(lib[param]["scale"])
                ax.set_title(param)
                ax.legend(fontsize = 10)
                ax.grid(which="major", alpha = 0.3)
                if param in ["NVi", "P", "M", "Ne", "Nn",  "Fcx", "Frec", "E", "F", "R", "Rex", "Rrec", "Riz", "Siz", "S", "Eiz", "Vi"]:
                    ax.set_xlim(9,10.2)
                ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1e}"))

def replace_guards(var):
    """
    This in-place replaces the points in the guard cells with the points on the boundary
    MK MODIFIED TO NOT BE IN PLACE
    """
    out = np.array(var)
    out[0] = 0.5*(out[0] + out[1])
    out[-1] = 0.5*(out[-1] + out[-2])
    
    return out

def constants(name):
    
    d = dict()
    d["mass_p"] = 1.6726219e-27 # Proton mass [kg]
    d["mass_e"] = 9.1093837e-31 # Electron mass [kg]
    d["a0"] = 5.29177e-11 # Bohr radius [m]
    d["q_e"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    d["k_b"] = 1.3806488e-23 # Boltzmann self.ant [JK^-1]
    
    return d[name]

def mike_cmap(number):
    colors = ["teal", "darkorange", "firebrick",  "limegreen", "magenta", "cyan", "navy"]
    return colors[:number]

def library():
    lib = defaultdict(dict)
    

    # Histories etc.
    lib["h_Ne"]["name"] = "History of upstream plasma density"
    lib["h_Ne"]["unit"] = "$m^{-3}$"
    
    # Others
    lib["NeSource"]["name"] = "Particle source"
    lib["NeSource"]["unit"] = "$m^{-3}s^{-1}$"
    lib["PeSource"]["name"] = "Energy source"
    lib["PeSource"]["unit"] = "$Wm^{-3}s^{-1}$"
    
    # Normalisation factors
    lib["Nnorm"]["name"] = "Density normalisation factor"
    lib["Nnorm"]["unit"] = "m-3"
    lib["Tnorm"]["name"] = "Temperature normalisation factor"
    lib["Tnorm"]["unit"] = "eV"
    lib["Cs0"]["name"] = "Speed normalisation factor"
    lib["Cs0"]["unit"] = "m/s"
    lib["Omega_ci"]["name"] = "Time normalisation factor"
    lib["Omega_ci"]["unit"] = "s-1"
    lib["rho_s0"]["name"] = "Length normalisation factor"
    lib["rho_s0"]["unit"] = "m"
    
    # Process parameters
    lib["P"]["name"] = "Plasma pressure"
    lib["P"]["unit"] = "Pa"
    lib["Te"]["name"] = "Plasma temperature"
    lib["Te"]["unit"] = "eV"
    lib["Pn"]["name"] = "Neutral pressure"
    lib["Pn"]["unit"] = "Pa"
    lib["dynamic_p"]["name"] = "Plasma dynamic pressure"
    lib["dynamic_p"]["unit"] = "Pa"
    lib["dynamic_n"]["name"] = "Neutral dynamic pressure"
    lib["dynamic_n"]["unit"] = "Pa"
    lib["Ne"]["name"] = "Plasma density"
    lib["Ne"]["unit"] = "$m^{-3}$"
    lib["Ne"]["scale"] = "linear"
    lib["Nn"]["name"] = "Neutral density"
    lib["Nn"]["unit"] = "$m^{-3}$"
    lib["Nn"]["scale"] = "linear"
    lib["Ntot"]["name"] = "Total density"
    lib["Ntot"]["unit"] = "$m^{-3}$"
    lib["Ntot"]["scale"] = "linear"
    lib["NVi"]["name"] = "Plasma flux"
    lib["NVi"]["unit"] = "$m^{-2}s^{-1}$"
    lib["NVi"]["scale"] = "linear"
    lib["Vi"]["name"] = "Ion velocity"
    lib["Vi"]["unit"] = "m/s"
    lib["Vi"]["scale"] = "linear"
    lib["NVn"]["name"] = "Neutral flux"
    lib["NVn"]["unit"] = "$m^{-2}s^{-1}$"
    lib["M"]["name"] = "Mach number"
    lib["M"]["unit"] = "-"
    lib["M"]["scale"] = "linear"
    lib["kappa_epar"]["name"] = "Plasma thermal conduction"
    lib["kappa_epar"]["unit"] = "?"
    lib["Dn"]["name"] = "Neutral diffusion coefficient"
    lib["Dn"]["unit"] = "$m^{2}s^{-1}$"
    lib["flux_ion"]["name"] = "Flux of ions to target"
    lib["flux_ion"]["unit"] = "?"

    # S: particle sink
    lib["S"]["name"] = "Sink of plasma density"
    lib["S"]["unit"] = "$m^{-3}s^{-1}$"
    lib["Srec"]["unit"] = "$m^{-3}s^{-1}$"
    lib["Srec"]["name"] = "Recomb. particle sink"
    lib["Srec"]["unit"] = "$m^{-3}s^{-1}$"
    lib["Siz"]["name"] = "Ioniz. particle sink"
    lib["Siz"]["unit"] = "$m^{-3}s^{-1}$"

    # F: momentum sink
    lib["F"]["name"] = "Sink of plasma momentum"
    lib["F"]["unit"] = "$Nm^{-3}$"
    lib["F"]["scale"] = "linear"
    lib["Frec"]["name"] = "Recomb F sink"
    lib["Frec"]["unit"] = "$Nm^{-3}$"
    lib["Frec"]["scale"] = "linear"
    lib["Fiz"]["name"] = "Ionisation F sink"
    lib["Fiz"]["unit"] = "$Nm^{-3}$"
    lib["Fcx"]["name"] = "Charge exch. F sink"
    lib["Fcx"]["unit"] = "$Nm^{-3}$"
    lib["Fcx"]["scale"] = "linear"
    lib["Fel"]["name"] = "Coll. F sink"
    lib["Fel"]["unit"] = "$Nm^{-3}$"

    # R: radiation sink
    lib["R"]["name"] = "Rad. energy sink"
    lib["R"]["unit"] = "$Wm^{-3}$"
    lib["Rrec"]["name"] = "Recomb. radiation loss"
    lib["Rrec"]["unit"] = "$Wm^{-3}$"
    lib["Riz"]["name"] = "13.6eV Ionisation energy sink"
    lib["Riz"]["unit"] = "$Wm^{-3}$"
    lib["Riz"]["scale"] = "linear"
    lib["Rzrad"]["name"] = "Radiation loss due to impurities"
    lib["Rzrad"]["unit"] = "$Wm^{-3}$"
    lib["Rex"]["name"] = "Radiation loss due to excitation"
    lib["Rex"]["unit"] = "$Wm^{-3}$"

    # E: energy sink
    lib["E"]["name"] = "Sink of plasma energy"
    lib["E"]["unit"] = "$Wm^{-3}$"
    lib["Erec"]["name"] = "Hvy particle exch. thermal energy sink, rec"
    lib["Erec"]["unit"] = "$Wm^{-3}$"
    lib["Eiz"]["name"] = "Hvy particle exch. thermal energy sink, IZ"
    lib["Eiz"]["unit"] = "$Wm^{-3}$"
    lib["Ecx"]["name"] = "Hvy particle exch. thermal energy sink, CX"
    lib["Ecx"]["unit"] = "$Wm^{-3}$"
    lib["Eel"]["name"] = "Sink of plasma energy due to elastic collisions"
    lib["Eel"]["unit"] = "$Wm^{-3}$"
    lib["Ert"]["name"] = "Sink of plasma energy due to Braginskii thermal friction"
    lib["Ert"]["unit"] = "$Wm^{-3}$" 
    
    for param in lib.keys():
        try:
            lib[param]["scale"]
        
        except:
            lib[param]["scale"] = "linear"
            
    return lib
    
def read_case(path_case, merge_output = False):
    """OLD DO NOT USE"""
    mode = platform.system()
    
    if mode == "Linux":
        case = path_case.split(r"/")[-1]
    elif mode == "Windows":
        case = path_case.split(r"\\")[-1]
        
    verbosity = False
    testmode = False
    tind = -1 #SELECTED TIME 
    # path_case = path(case)

    """
    READING
    """
    # The array format is [t, x, y, z]. In 1D, the length is in the y direction.
    # Along the length, there are N points. On each end, there are two guard cells.
    # SD1D only uses one of them. This is why we need to extract 1:-1 (from second to second to last) values.
    # The BCs are halfway between the guard cell and the final grid cell. This is not very practical.
    # The replace_guard function is therefore used to calculate what the BC values should be and inserts those
    # into the guard cells. After guard replacement, the first cell is the plasma BC, and the final the wall BC.

    print(f"Reading case {case} || ", end="")

    result = dict()
    vars_missing = list()

    # Collect variables
    var_collect = ["P", "Ne", "Nn", "NVi", "NVn", "kappa_epar", "Pn", "Dn",
                  "S", "Srec", "Siz", # Plasma particles and sinks
                  "F", "Frec", "Fiz", "Fcx", "Fel", # Momentum source/sinks to neutrals
                  "R", "Rrec", "Riz", "Rzrad", "Rex", # Radiation, energy loss from system
                  "E", "Erec", "Eiz", "Ecx", "Eel", # Energy transfer between neutrals and plasma
                  "Pe", "Pd+", "Ve", "Vd+", "Nd", "Nd+", "Td+", "SPd+", "SNd+", "Ert"]
                 

    for var in var_collect:
        try:
            result[var] = collect(var, path=path_case, yguards=True, info = verbosity, strict = True)[-1,0,1:-1,0]
        except:
            vars_missing.append(var)
            pass

    print(vars_missing)
        
    if "Vd+" in vars_missing:
        hermes = False
    else:
        hermes = True

    # Collect constants
    const_collect = ["Nnorm", "Tnorm", "Cs0", "Omega_ci", "rho_s0"]
    const = dict()
    for var in const_collect:
        try:
            const[var] = collect(var, path = path_case, info = verbosity, strict = True)
        except:
            vars_missing.append(var)
            pass

    # Collect sources
    source_collect = ["PeSource", "NeSource"]
    
    try:
        result["PeSource"] = collect(
                "PeSource", path = path_case, yguards= True, info = verbosity, strict = True)[-1,1:-1]
    except:
        vars_missing.append("PeSource")
    
    try:
        result["NeSource"] = collect(
            "NeSource", path = path_case, yguards= True, info = verbosity, strict = True)[-1,0,1:-1]
    except:
        vars_missing.append("NeSource")

    # Collect mesh information
    mesh_collect = ["dy", "J", "g_22"]
    for var in mesh_collect:
        try:
            result[var] = collect(var, path=path_case, yguards=True, info = verbosity, strict = True)[0,1:-1]
        except:
            vars_missing.append(var)

    print("Missing vars: ", end = "")
    [print("-"+x, end = " ") for x in vars_missing]
    print("||")
    if len(vars_missing) == 0:
        print("...None")

    else:
        for var in vars_missing:
            result[var] = np.zeros_like(result["Ne"])

    # Calculate temperature
    result["Te"] = (0.5 * result["P"] / result["Ne"] ) # Electron temp
    
    if "Pe" not in vars_missing:
        result["Te"] = (result["Pe"] / result["Ne"] ) # Electron temp
    
    # Derive SD1D-like variables
    if hermes:
        result["Nn"] = result["Nd"]
        result["Vi"] = result["Vd+"]
        result["NVi"] = result["Vd+"] * result["Nd+"]

    normalised = False # Flag for normalisation section
    guardreplaced = False # Flag for guard replacement section

    # Guard replacement
    if guardreplaced == False:
        for var in result.keys():
            
            if var not in ["NVi" \
                          ,"S", "Srec", "Siz", # Plasma particles and sinks
                           "F", "Frec", "Fiz", "Fcx", "Fel", # Momentum source/sinks to neutrals
                           "R", "Rrec", "Riz", "Rzrad", "Rex", # Radiation, energy loss from system
                           "E", "Erec", "Eiz", "Ecx", "Eel", "Ert", # Energy transfer between neutrals and plasma
                           "Dn", "SNd+", "SPd+",
                  ]: # NVi already has its last value as the sheath flux.

                result[var] = replace_guards(result[var])
                
            # There are no sources in guard cells, so let's set them to NaN so they can be plotted
            # against a grid that includes the guard cells.
            elif var not in ["NVi"]:
                result[var][-1] = np.nan
                result[var][0] = np.nan
                
    # Correct Te
    result["Te"][-1] = result["Te"][-2]
            
    """
    Nnorm : Density : [m-3]
    Tnorm : Temperature : [eV]
    Pnorm : Pressure : [Pa]
    Snorm : Density flux : [m-3s-1]
    Fnorm : Momentum : [kgm-2s-2, or Nm-3]
    Enorm : Power : [Wm-3]
    Cs0 : Speed : [ms-1]
    Omega_ci : Time : [s-1]
    rho_s0 : Length : [m]
    """

    # Quantities derived pre-normalisation
    i_charge = 1.602e-19
    i_mass = 1.67272e-27*2
    const["Pnorm"] = const["Nnorm"] * const["Tnorm"] * i_charge # Converts P to Pascals. 1.602e-19 is proton charge in C
    const["Snorm"] = const["Nnorm"] * const["Omega_ci"] # Normalisation for S: plasma density sink (m-3s-1)
    const["Fnorm"] = i_mass * const["Nnorm"] * const["Cs0"] * const["Omega_ci"] # Plasma momentum sink normalisation (kgm-2s-1)
    const["Enorm"] = i_charge * const["Nnorm"] * const["Tnorm"] * const["Omega_ci"] # Plasma energy sink normalisation
    
    # Denormalisation
    list_tnorm = ["Te"] # [eV]
    list_nnorm = ["Ne", "Nn", "Nd+", "Nd"] # [m-3]
    list_pnorm = ["P", "Pn", "dynamic_p", "dynamic_n", "Pe", "SPd+"] # [Pa]
    list_snorm = ["S", "Srec", "Siz", "NeSource", "SNd+"] # [m-3s-1]
    list_fnorm = ["F", "Frec", "Fiz", "Fcx", "Fel"] # [kgm-2s-2 or Nm-3]
    list_enorm = ["E", "R", "Rrec", "Riz", "Rzrad", "Rex", "Erec", "Eiz", "Ecx", "Eel", "Ert"] # Wm-3
    
    result_dn = dict() # Place for denormalised variables

    try:
        result_dn["PeSource"] = result["PeSource"] * const["Pnorm"] * const["Omega_ci"] # [Pa/s]
    except:
        pass

    for var in result:
        if var in list_tnorm:
            result_dn[var] = result[var] * const["Tnorm"]
        elif var in list_nnorm:
            result_dn[var] = result[var] * const["Nnorm"]
        elif var in list_pnorm:
            result_dn[var] = result[var] * const["Pnorm"]
        elif var in list_snorm:
            result_dn[var] = result[var] * const["Snorm"]
        elif var in list_fnorm:
            result_dn[var] = result[var] * const["Fnorm"]
        elif var in list_enorm:
            result_dn[var] = result[var] * const["Enorm"]
            
    # Other denormalisations
    result_dn["Dn"] = result["Dn"] * const["rho_s0"] * const["rho_s0"] * const["Omega_ci"] # [m][m][s-1]=[m^2/s] diffusion coefficient
    result_dn["Ve"] = result["Ve"] * const["Cs0"]
    result_dn["Vd+"] = result["Vd+"] * const["Cs0"]
    result_dn["Td+"] = result["Td+"] * const["Tnorm"]
    

    """
    DERIVED QUANTITIES, DENORMALISATION, GUARD REPLACEMENT
    """

    # Quantities derived post-denormalisation
    result_dn["Ntot"] = result_dn["Ne"] + result_dn["Nn"]
    result_dn["NVi"] = result["NVi"] * const["Nnorm"] * const["Cs0"]
    result_dn["Vi"] = (result_dn["NVi"] / result_dn["Ne"])  # Ion parallel velocity
    result_dn["Cs"] = const["Cs0"] * np.sqrt(2 * result_dn["Te"]/const["Tnorm"]) # Sound speed
    result_dn["M"] = result_dn["Vi"] / result_dn["Cs"]
    result_dn["dynamic_p"] = result["NVi"]**2 / result["Ne"] * const["Pnorm"]
    result_dn["ESource"] = result_dn["PeSource"] * 3/2 # Is this right?
    result_dn["SEd+"] = result_dn["SPd+"] * 3/2
    
    result_dn["flux_out"] = (result["NVi"][-1] * result["J"] / np.sqrt(result["g_22"]) * const["Omega_ci"] * const["Nnorm"])[0] # Sheath flux in [s-1]. J/g_22 is cross-sectional area (J*dy)/(sqrt(g_22)*dy) = Volume/length

    try:
        result_dn["dynamic_n"] = result["NVn"]**2 / result["Nn"] * const["Pnorm"]
    except:
        pass


    n = len(result["dy"])
    result_dn["pos"] = np.zeros(n)
    result_dn["pos"][0] = -0.5*result["dy"][1]
    result_dn["pos"][1] = 0.5*result["dy"][1]

    for i in range(2,n):
        result_dn["pos"][i] = result_dn["pos"][i-1] + 0.5*result["dy"][i-1] + 0.5*result["dy"][i]

    # replace guards for pos
    result_dn["pos"] = replace_guards(result_dn["pos"])
    
    # Add dy and J, calculate dV
    result_dn["dy"] = result["dy"]
    result_dn["J"] = result["J"]
    result_dn["dV"] = result_dn["dy"] * result_dn["J"]
    
    """
    OTHER QUANTITIES
    """
    # Line averaged density

    result_dn["Ne_avg"] = sum(result_dn["Ne"] * result_dn["dV"])/sum(result_dn["dy"])
    result_dn["Nn_avg"] = sum(result_dn["Nn"] * result_dn["dV"])/sum(result_dn["dy"])
    result_dn["Ntot_avg"] = result_dn["Ne_avg"] + result_dn["Nn_avg"]
    
    # Power integrals
    

    """
    COLLECTION
    """

    case = dict()
    case["result"] = result_dn
    case["const"] = const
    # print(len(case["result"]["R"]))
    
    if merge_output:
        case = {**case["result"], **case["const"]}
        
    return case


