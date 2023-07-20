#!/usr/bin/env python3

from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import traceback
import platform
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
        self.boutdata = BoutData(self.casepath, yguards=True, strict = True, DataFileCaching=False)
        self.raw_data = self.boutdata["outputs"]
        self.options = self.boutdata["options"]

        self.get_timestats()

    def extract_variables(self, tind = -1):
        self.missing_vars = []
        self.norm_data = dict()

        # NOTE: in Hermes-3, NVi is actually the momentum in [kg m-2 s-1].
        # That means to get flux you need to do NVi/(mass_p * AA).
        # Which means you can get flux by normalising NVd+ by 1/AA * Nnorm * Cs0 to get [m-2s-1].

        var_collect = ["P", "Ne", "Nn", "NVi", "NVn", "NVd", "NVd+", "kappa_epar", "Pn", "Dn",
                        "S", "Srec", "Siz", 
                        "F", "Frec", "Fiz", "Fcx", "Fel", # Momentum source/sinks to neutrals
                        "R", "Rrec", "Riz", "Rzrad", "Rex", # Radiation, energy loss from system
                        "E", "Erec", "Eiz", "Ecx", "Eel", # Energy transfer between neutrals and plasma
                        "Pe", "Pd", "Pd+", "Ve", "Vd+", "Vi", "Nd", "Nd+", "Td+", "SPd+", "SNd+", "Ert", "Te",
                        "PeSource", "NeSource",
                        "Div_Q_SH", "Div_Q_SNB",
                        # Hermes only variables
                        "Dd_Dpar",
                        "Sd_Dpar", "Sd+_iz", "Sd+_rec",
                        "Ed_Dpar", "Edd+_cx", "Ed+_iz", "Ed+_rec", "Ed+_src", "Ee_src",
                        "Fd_Dpar", "Fdd+_cx", "Fd+_iz", "Fd+_rec",
                        "Rd+_ex", "Rd+_rec"
                        ]

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

        if "Pd+" in self.missing_vars:
            self.ion_eqn = True
        else:
            self.ion_eqn = False

        if "NVn" not in self.missing_vars or "NVd" not in self.missing_vars:
            self.evolve_nvn = True
        else:
            self.evolve_nvn = False

        if "Pn" not in self.missing_vars or "Pd" not in self.missing_vars:
            self.evolve_pn = True
        else:
            self.evolve_pn = False

        if "Div_Q_SNB" not in self.missing_vars:
            self.snb = True
        else:
            self.snb = False

        #------------Derive variables
        # Hermes-3 deals with it differently and the raw J is equal to Bnorm/Lnorm = 1/rho_s0
        # Only relative changes are important... so let's just normalise it like this so 
        # that it's consistent with SD1D. TODO figure this out properly
        """
        Hey @mikekryjak It's the momentum density, rather than the particle flux. Dividing by AA gives the particle flux as it was in SD1D.
        For example, the NVd+ output if multiplied by m_p * Nnorm * Cs0 is the momentum density, or equivalently mass flux, with units of [kg m^-2 s^-1]. m_p here is the proton mass.
        Taking out the AA factor gives particle flux, so NVd+ / 2 multiplied by Nnorm * Cs0 is the particle flux with units of [m^-2 s^-1]
        
        """
        self.J = self.J / min(self.J)
        self.dV = self.dy * self.J
        
        if self.hermes:
            self.norm_data["Nn"] = self.norm_data["Nd"]
            self.norm_data["Vi"] = self.norm_data["Vd+"]
            self.norm_data["NVi"] = self.norm_data["NVd+"] * constants("mass_p") / (constants("mass_p")*2) # Originally [kgm2s-1], then divide by mass_i to get flux.
            if "Pd+" in self.norm_data.keys():
                self.norm_data["P"] = self.norm_data["Pe"] + self.norm_data["Pd+"]
            else:
                self.norm_data["P"] = self.norm_data["Pe"]
            self.norm_data["S"] = self.norm_data["SNd+"]
            
            if "Pd+" in self.norm_data.keys():
                self.norm_data["Ti"] = self.norm_data["Td+"]
            
            if "Rd+_rec" in self.norm_data.keys():
                self.norm_data["Rex"] = self.norm_data["Rd+_ex"]
                self.norm_data["Rrec"] = self.norm_data["Rd+_rec"]
                self.norm_data["R"] = self.norm_data["Rex"] + self.norm_data["Rrec"]
                self.norm_data["Srec"] = self.norm_data["Sd+_rec"]
                self.norm_data["Siz"] = self.norm_data["Sd+_iz"]
                self.norm_data["Fiz"] = self.norm_data["Fd+_iz"]
                self.norm_data["Frec"] = self.norm_data["Fd+_rec"]
                self.norm_data["Eiz"] = self.norm_data["Ed+_iz"]
                self.norm_data["Erec"] = self.norm_data["Ed+_rec"]

            if "Dd_Dpar" in self.norm_data.keys():
                self.norm_data["Dn"] = self.norm_data["Dd_Dpar"]
            if "Fdd+_cx" in self.norm_data.keys():
                self.norm_data["Fcx"] = self.norm_data["Fdd+_cx"]
                self.norm_data["Ecx"] = self.norm_data["Edd+_cx"]
            self.dneut = self.options["neutral_parallel_diffusion"]["dneut"]
        else:
            # self.norm_data["Vi"] = self.norm_data["NVi"] / self.norm_data["Ne"] # in SD1D AA is in normalisation update: this is already in file...
            self.dneut = self.options["sd1d"]["dneut"]
            

        if self.evolve_nvn:
            if self.hermes:
                self.norm_data["NVn"] = self.norm_data["NVd"] * constants("mass_p") / (constants("mass_p")*2) # Originally [kgm2s-1], then divide by mass_i to get flux.
                self.norm_data["Vd"] = self.norm_data["NVd"] / self.norm_data["Nd"] # AA
                self.norm_data["Vn"] = self.norm_data["Vd"]

        if self.evolve_pn:
            if self.hermes:
                self.norm_data["Tn"] = self.norm_data["Pd"] / self.norm_data["Nd"] 
                self.norm_data["Pn"] = self.norm_data["Pd"]
            else:
                self.norm_data["Tn"] = self.norm_data["Pn"] / self.norm_data["Nn"]
            
        if self.ion_eqn:
            self.norm_data["Te"] = (self.norm_data["Pe"] / self.norm_data["Ne"] ) # Electron temp
            self.norm_data["Ni"] = self.norm_data["Nd+"]
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
        Xnorm : Flux [m2s-1]
        Qnorm : Heat flux [Wm-2]
        Cs0 : Speed : [ms-1]
        Omega_ci : Time : [s-1]
        rho_s0 : Length : [m]

        CURRENT = Nnorm * Cs0 = [m-3s-1]. Because of Z*Ni*Vi and Z is unitless
        POTENTIAL = Tnorm = [eV] = [C] remember 1eV = q_e [J] = q_e [C]

        """

        # Derived normalisation factors
        self.Pnorm = self.Nnorm * self.Tnorm * constants("q_e") # Converts P to Pascals. 1.602e-19 is proton charge in C
        self.Snorm = self.Nnorm * self.Omega_ci # Normalisation for S: plasma density sink (m-3s-1)
        if self.hermes:
            self.Fnorm = (constants("mass_p")) * self.Nnorm * self.Cs0 * self.Omega_ci # Everything normalised to mass_p not mass_i in Hermes-3
        else:
            self.Fnorm = (constants("mass_p") * 2) * self.Nnorm * self.Cs0 * self.Omega_ci # [kgm-2s-1] Plasma momentum sink normalisation
        self.Enorm = constants("q_e") * self.Nnorm * self.Tnorm * self.Omega_ci # [Wm-3] Plasma energy sink normalisation
        self.Vnorm = self.Cs0 # [ms-1] Velocity
        self.Xnorm = self.Nnorm * self.Cs0 # [m-2 s-1] Flux
        self.Dnorm = self.rho_s0 * self.rho_s0 * self.Omega_ci  # [m2s-1] Diffusion
        self.Qnorm = self.Enorm * self.rho_s0 # [Wm-2] Heat flux
        
        list_tnorm = ["Te", "Td+", "Ti", "Td", "Tn"] # [eV]
        list_nnorm = ["Ne", "Nn", "Nd+", "Nd", "Ni"] # [m-3]
        list_pnorm = ["P", "Pn", "dynamic_p", "dynamic_n", "Pe", "SPd+", "Pd+", "Pd"] # [Pa]
        list_snorm = ["S", "Srec", "Siz", "NeSource", "SNd+", "Sd_Dpar", "Siz", "Srec",
                    "Sd+_iz", "Sd+_rec"] # [m-3s-1]
        list_fnorm = ["F", "Frec", "Fiz", "Fcx", "Fel", "Fd_Dpar", "Fdd+_cx"] # [kgm-2s-2 or Nm-3]
        list_enorm = ["E", "R", "Rrec", "Riz", "Rzrad", "Rex", "Erec", 
                    "Eiz", "Ecx", "Eel", "Ert", "PeSource",
                    "Div_Q_SH", "Div_Q_SNB",
                    "Ed_Dpar", "Edd+_cx","Rd+_ex",
                    "Ed+_src", "Ee_src"] # [Wm-3]
        list_vnorm = ["Vi", "Ve", "Vd+", "Vd", "Vn"] 
        list_xnorm = ["NVi", "NVn", "NVd", "NVd+"] # m-2s-1
        list_dnorm = ["Dn", "Dd_Dpar"] # m2s-1
        list_qnorm = [] # [Wm-2]

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
            elif var in list_vnorm:
                dnorm[var] = norm[var] * self.Vnorm
            elif var in list_xnorm:
                dnorm[var] = norm[var] * self.Xnorm
            elif var in list_dnorm:
                dnorm[var] = norm[var] * self.Dnorm
            elif var in list_qnorm:
                dnorm[var] = norm[var] * self.Qnorm

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
                    
        # Correct Te to reflect BC condition within code which is zero gradient
        if self.hermes:
            dnorm["Te"][-1] = dnorm["Te"][-2]

        #------------Derived variables
        dnorm["Ntot"] = dnorm["Ne"] + dnorm["Nn"]
        # dnorm["Cs"] = self.Cs0 * np.sqrt(2 * dnorm["Te"]/self.Tnorm) # Sound speed
        dnorm["Cs"] = np.sqrt(constants("q_e") * (dnorm["Te"]*2)/(constants("mass_p")*2))
        dnorm["M"] = dnorm["Vi"] / dnorm["Cs"]
        dnorm["dynamic_p"] = norm["NVi"]**2 / norm["Ne"] * self.Pnorm
        dnorm["Ne_avg"] = sum(dnorm["Ne"] * self.dV)/sum(self.dy)
        dnorm["Nn_avg"] = sum(dnorm["Nn"] * self.dV)/sum(self.dy)
        dnorm["Ntot_avg"] = dnorm["Ne_avg"] + dnorm["Nn_avg"]

        if self.hermes and "Pd+" in self.norm_data.keys():
            dnorm["SEd+"] = dnorm["SPd+"] * 3/2

        # else:
        #     dnorm["ESource"] = dnorm["PeSource"] * 3/2 # Is this right?

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
        
        
        self.options["NOUT"]
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

    def plot_density_controller(
        self, plot_start = 0, plot_end = -1, 
                                zoom = 1, ymin = None, ymax = None,
                            labels_override = None, absolute_source = False, 
                                error_plot = False, single_plot = True):
        # Plot the history of upstream density to show
        # performance of PI controller.
        # Zoom: controls y axis limits.
        # Stepsize: controls how many timepoints are sampled to reduce loading time.

        plot_cases = [self.casepath]

        # Collect density history
        h_Ne = defaultdict(list)
        h_NeSource = defaultdict(list)
        intended_Ne = dict() # Intended upstream density.
        num_timesteps = dict()
        error_Ne = dict()
        labels = dict()
        final_values = list() # for finding final values for plot scales

        print("Collecting data...", end = "")
        for i, case in enumerate(plot_cases):
        # Collect upstream density (density in first cell) for each time point.

            # Extract number of timesteps in total.
            
            Nnorm = collect("Nnorm", path = self.casepath)
            Omega_ci = collect("Omega_ci", path = self.casepath)
            hist_Ne = collect("Ne", path = self.casepath, yguards=True, info=False)[:,0,1:-1,0]
            
            try:
                if self.hermes:
                    hist_NeSource = collect("Sd+_src", path = self.casepath, yguards=True, info=False)[:,0,1:-1]
                else:
                    hist_NeSource = collect("NeSource", path = self.casepath, yguards=True, info=False)[:,0,1:-1]
            except:
                print("NeSource not found for case {}".format(case))
                pass
            num_timesteps[case] = hist_Ne.shape[0]

            for t in range(num_timesteps[case]):
                h_Ne[case].append(replace_guards(hist_Ne[t,:])[0]*Nnorm)
                
                try:
                    h_NeSource[case].append(replace_guards(hist_NeSource[t,:])[0]*Nnorm*Omega_ci)
                except:
                    pass

            if self.hermes:
                intended_Ne[case] = BoutData(self.casepath)["options"]["d+"]["density_upstream"]
            else:
                intended_Ne[case] = BoutData(self.casepath)["options"]["sd1d"]["density_upstream"]
            
            
            # Find Ne error
            error_Ne[case] = abs(1 - np.array(h_Ne[case]) / intended_Ne[case])
            
            # Append final values
            if error_plot == True:
                final_values.append(error_Ne[case][-1])
            else:
                final_values.append(h_Ne[case][-1])

            # Get labels done
            if labels_override != None:
                labels[case] = labels_override[i]
            else:
                labels[case] = case
                
            # print(intended_Ne)
            # print(error_Ne)
                

            print("{}...".format(case), end="")
        print("\nComplete.")

        # Make colors
        colors = dict(zip(plot_cases, cc.glasbey_category10[:len(plot_cases)]))

        # Plot
        if single_plot == True:
            for case in plot_cases:
                fig, ax = plt.subplots(figsize = [6,6])
                ax2 = ax.twinx()
                ax.set_title("Upstream density error")
                ax.plot(range(len(h_Ne[case])), 
                        h_Ne[case], label = "Nu", 
                        linewidth = 1, color = "black")
                ax.hlines(intended_Ne[case], 0, num_timesteps[case], linewidth = 2,
                        linestyle = "--", color = "black", label = "Nu target")  
                
                ax.grid(which="both")
                ax2.plot(range(len(h_NeSource[case])), 
                        h_NeSource[case], 
                        linewidth = 2, color = "red", alpha = 1, label = "Particle source")
                ax.set_ylabel("Upstream density (m-3)")
                ax2.set_ylabel("Upstream particle source (s-1)")
                ax.set_xlabel("Timestep")
                fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes, fontsize = 12)
            
        else:  
            fig, ax = plt.subplots(1,2, figsize = [12,6])
            for case in plot_cases:   
                if error_plot == True: 
                    ax[0].set_title("Upstream density error")
                    ax[0].plot(range(len(error_Ne[case])), 
                            error_Ne[case], label = f"{labels[case]}", 
                            linewidth = 1, color = colors[case])

                else:
                    ax[0].set_title("Upstream density & target")
                    ax[0].plot(range(len(h_Ne[case])), 
                            h_Ne[case], label = f"{labels[case]}", 
                            linewidth = 2, color = colors[case], linestyle = "-")

                    ax[0].hlines(intended_Ne[case], 0, num_timesteps[case], linewidth = 2,
                            linestyle = "--", color = colors[case])     
                try:
                    if absolute_source == True:
                        ax[1].plot(range(len(h_NeSource[case])), 
                                abs(np.array(h_NeSource[case])), 
                                label = f"{labels[case]}", 
                                linewidth = 2, color = colors[case])
                    else:
                        ax[1].plot(range(len(h_NeSource[case])), 
                                h_NeSource[case], label = f"{labels[case]}", 
                                linewidth = 2, color = colors[case])
                except:
                    pass

    def get_htransfer(self):
        
        self.data["pos_boundary"], self.data["convect_ke"] = ke_convection(self.pos, self.data["Ne"], self.data["Vi"])
        _, self.data["convect_therm"] = ke_convection(self.pos, self.data["P"], self.data["Vi"])

        # if self.snb:
        #     self.data["conduction"] = self.data["Div_Q_SNB"]
        #     self.data["conduction_SH"] = self.data["Div_Q_SH"]
            
        # else:
        #     _, self.data["conduction"] = heat_conduction(self.pos, self.data["Te"])
        #     self.data["conduction_SH"] = self.data["conduction"]

        _, self.data["conduction_SH_script"] = heat_conduction(self.pos, self.data["Te"])
        
    def heat_balance(self, verbose = False, override = False):
        if self.hermes and override == False:
            print("Hermes heat balance not implemented")
        else:

            self.sheath_gamma = self.options["sd1d"]["sheath_gamma"]
            self.flux_sheath = self.data["NVi"][-1] # Not correct in Hermes
            self.Tt = self.data["Te"][-1]
            
            self.hflux_source = np.nansum(self.data["ESource"] * self.dV) * 1e-6
            self.hflux_sheath = self.Tt * constants("q_e") * self.flux_sheath * self.sheath_gamma * 1e-6

            self.hflux_E = np.trapz((self.data["E"] * self.dV)[1:-1]) * 1e-6
            self.hflux_R = np.trapz((self.data["R"] * self.dV)[1:-1]) * 1e-6
            self.hflux_F = np.trapz((self.data["F"][1:-1] * self.data["Vi"][1:-1] * self.dV[1:-1])) * 1e-6

            self.hflux_out = self.hflux_sheath + self.hflux_E + self.hflux_R + self.hflux_F
            self.hflux_imbalance = self.hflux_out - self.hflux_source
            self.hflux_imbalance_ratio = self.hflux_imbalance / self.hflux_source

            if self.snb:
                q_snb = self.raw_data["Div_Q_SNB"].squeeze()[-1,:]
                q_snb_dn = q_snb * self.Qnorm
                q_sh = self.raw_data["Div_Q_SH"].squeeze()[-1,:]
                q_sh_dn = q_sh * self.Qnorm


                hflux_last_snb = q_snb_dn[-3] * 2 * 1e-6
                hflux_last_snb_gr = np.mean([q_snb_dn[-3], q_snb_dn[-2]]) * 2 * 1e-6

                hflux_last_sh = q_sh_dn[-3] * 2 * 1e-6
                hflux_last_sh_gr = np.mean([q_sh_dn[-3], q_sh_dn[-2]]) * 2 * 1e-6

                self.hflux_lastcell_snb = hflux_last_snb
                self.hflux_lastcell_sh = hflux_last_sh

            if verbose:
                print(">> Fluxes (MW):")
                print(f"- Losses to neutrals:- {self.hflux_E:.3f}[MW]")
                print(f"- Radiation:---------- {self.hflux_R:.3f}[MW]")
                print(f"- Friction losses:---- {self.hflux_F:.3f}[MW]")
                print(f"- Sheath:------------- {self.hflux_sheath:.3f}[MW]")
                print(f"ooooooooooooooooooooooooooooooooo")
                print(f"- Total out:---------- {self.hflux_out:.3f}[MW]")
                print(f"- Total in:----------- {self.hflux_source:.3f}[MW]")
                print(f"- Difference:--------- {self.hflux_imbalance:.3f}[MW]")
                print(f"- Imbalance:---------- {self.hflux_imbalance_ratio:.1%}")

    def heat_balance_hermes(self):
        """
        This works only for a single ion species plasma
        """

        d = BoutData(self.casepath, yguards = True, info = False, strict = True, DataFileCaching=False)["outputs"]
        o = BoutData(self.casepath, yguards = True, info = False, strict = True, DataFileCaching=False)["options"]


        def get_boundary(x):
            return (x[-2] + x[-3])/2

        def extract(x):
            return x.squeeze()[tind,:]

        def volume_integral(x):
            return np.trapz(x = pos[2:-2], y = x[2:-2] * J[2:-2])

        tind = -1

        gamma_e = float(o["sheath_boundary_simple"]["gamma_e"])
        gamma_i = float(o["sheath_boundary_simple"]["gamma_i"])

        try:
            Ge = float(o["sheath_boundary_simple"]["secondary_electron_coef"])
        except:
            Ge = 0 # Default
            pass

        try:
            polytropic = float(o["sheath_ion_polytropic"])
        except:
            polytropic = 1 # Default
            pass

        mass_p = constants("mass_p")
        mass_i = constants("mass_p") * 2 # [kg]
        mass_e = constants("mass_e") # [kg]
        q_e = constants("q_e") # [J/eV]
        k_b = constants("k_b") # [J/K]

        Vnorm = d["Cs0"]
        Enorm = q_e * d["Nnorm"] * d["Tnorm"] * d["Omega_ci"] # [Wm-3]
        Xnorm = d["Cs0"] * d["Nnorm"] # [m-2s-1]
        Nnorm = d["Nnorm"] # [m-3]
        Tnorm = d["Tnorm"] # [eV]
        Bnorm = d["Bnorm"] # [T]
        Pnorm = Nnorm * Tnorm * q_e # [Pa]
        Fnorm = mass_i * Nnorm * d["Cs0"] * d["Omega_ci"]


        # TODO include gamma_n (which is not implemented yet)

        # ----- Unpack things
        # NOTE here NVi is just flux in [m2s-1] while in hermes it's technically kgm2s-1
        Zi = 1
        AA = 2
        NVi = extract(d["NVd+"]) * Xnorm * mass_p / mass_i # Originally [kgm2s-1], then divide by mass_i to get flux.
        Ve = extract(d["Ve"]) * Vnorm
        Vi = extract(d["Vd+"]) * Vnorm
        Ti = extract(d["Td+"]) * Tnorm
        Te = extract(d["Te"]) * Tnorm
        Ni = extract(d["Nd+"]) * Nnorm
        Ne = extract(d["Ne"]) * Nnorm
        Pe = extract(d["Pe"]) * Pnorm
        Pi = extract(d["Pd+"]) * Pnorm
        Fcx = extract(d["Fdd+_cx"]) * Fnorm
        Rrec = extract(d["Rd+_rec"]) * Enorm
        Rex = extract(d["Rd+_ex"]) * Enorm
        Ee_src = extract(d["Pe_src"]) * Enorm * 3/2
        Ei_src = extract(d["Pd+_src"]) * Enorm * 3/2
        R = Rrec + Rex

        # ----- Geometry
        J = d["J"].squeeze() * d["rho_s0"] / Bnorm # from hermes-3.cxx
        g_22 = d["g_22"].squeeze() * d["rho_s0"]**2 # from hermes-3.cxx
        dy = d["dy"].squeeze()
        dV = J * dy
        A = get_boundary(J) / np.sqrt(get_boundary(g_22)) # sheath area

        # ------ Reconstruct grid position from dy
        n = len(dy)
        pos = np.zeros(n)
        pos[0] = -0.5*dy[1]
        pos[1] = 0.5*dy[1]

        for i in range(2,n):
            pos[i] = pos[i-1] + 0.5*dy[i-1] + 0.5*dy[i]


        Ti_t = get_boundary(Ti)
        Te_t = get_boundary(Te)

        # From saved output:
        case_i_hflux_add = -1 * d["Ed+_sheath"].squeeze()[-1,:][-3] * dV[-3] * Enorm * 1e-6
        case_e_hflux_add = -1 * d["Ee_sheath"].squeeze()[-1,:][-3] * dV[-3] * Enorm * 1e-6

        Cs = np.sqrt(q_e * (polytropic*Ti_t + Zi*Te_t)/mass_i)

        # ----- Boundary particle flux

        sheath_i_flux = -get_boundary(Ni) * get_boundary(Vi) * A # [s-1]
        sheath_e_flux = -get_boundary(Ne) * get_boundary(Ve) * A # [s-1]

        # ----- Boundary heat flux for ions

        sheath_i_hflux_th = 2.5 * Ti_t*q_e * sheath_i_flux * 1e-6 # Stangeby
        sheath_i_hflux_ke = 0.5 * mass_i * Cs**2 * sheath_i_flux * 1e-6 # Stangeby
        sheath_i_hflux_total = sheath_i_hflux_th + sheath_i_hflux_ke

        sheath_i_hflux_desired = gamma_i * Ti_t * q_e * sheath_i_flux * 1e-6
        sheath_i_hflux_additional = sheath_i_hflux_desired - sheath_i_hflux_total

        # ----- Boundary heat flux for electrons

        sheath_e_hflux_th = 2.5 * Te_t*q_e * sheath_e_flux * 1e-6 # Stangeby
        sheath_e_hflux_ke = 0.5 * mass_e * Cs**2 * sheath_e_flux * 1e-6 # Stangeby
        sheath_e_hflux_total = sheath_e_hflux_th + sheath_e_hflux_ke

        sheath_e_hflux_desired = gamma_e * Te_t * q_e * sheath_e_flux * 1e-6
        sheath_e_hflux_additional = sheath_e_hflux_desired - sheath_e_hflux_total

        # ----- Total
        sheath_total_hflux = sheath_i_hflux_total + sheath_e_hflux_total

        # ----- Radiation
        Rex_hflux = volume_integral(Rex) * 1e-6
        Rrec_hflux = volume_integral(Rrec) * 1e-6

        # ----- Source
        src_e_hflux = volume_integral(Ee_src) * 1e-6
        src_i_hflux = volume_integral(Ei_src) * 1e-6

        total_in = src_e_hflux + src_i_hflux
        total_out = Rex_hflux + Rrec_hflux + sheath_total_hflux

        print("\n- Ions ---------------")
        print(f"- Required additional power for chosen gamma = {sheath_i_hflux_additional:.2f} [MW]")
        print(f"- Additional cooling power in code = {case_i_hflux_add:.2f} [MW]")
        print("\n- Electrons ---------------")
        print(f"- Required additional power for chosen gamma = {sheath_e_hflux_additional:.2f} [MW]")
        print(f"- Additional cooling power in code = {case_e_hflux_add:.2f} [MW]")

        print("\n- All source summary ---------------")
        print(f"- Electron energy source = {src_e_hflux:.2f} [MW]")
        print(f"- Ion energy source = {src_i_hflux:.2f} [MW]")

        print(f"- Sheath ion heat flux = {sheath_i_hflux_desired:.2f} [MW]")
        print(f"- Sheath electron flux = {sheath_e_hflux_desired:.2f} [MW]")
        print(f"- Excitation = {Rex_hflux:.2f} [MW]")
        print(f"- Recombination = {Rrec_hflux:.2f} [MW]")

        print("\n- Totals ---------------")
        print(f"- Total in = {total_in:.2f} [MW]")
        print(f"- Total out = {total_out:.2f} [MW]")
        print(f"- Difference = {total_in + total_out:.2f} [MW] or {(total_in+total_out)/total_in:.1%}")

    def mass_balance_hermes(self):
        """
        Perform a total (system, so ions and neutrals) mass balance 
        for Hermes-3. This is done from scratch and only requires BoutData.
        TODO: Figure out if nloss exists in Hermes-3 and include it
        """

        # Collect all guard cells, so 2 on each side. The outer ones are not used. 
        # Final domain grid centre is therefore index [-3].
        # J/sqrt(g_22) = cross-sectional area of cell

        boutdata = BoutData(self.casepath, yguards = True, info = False, strict = True, DataFileCaching=False)
        d = boutdata["outputs"]
        o = boutdata["options"]
        tind = -1
        J = d["J"].squeeze()
        dy = d["dy"].squeeze()
        dV = J * dy
        g_22 = d["g_22"].squeeze()

        mass_p = constants("mass_p")
        mass_i = mass_p * 2

        # Reconstruct grid position from dy
        n = len(dy)
        pos = np.zeros(n)
        pos[0] = -0.5*dy[1]
        pos[1] = 0.5*dy[1]

        for i in range(2,n):
            pos[i] = pos[i-1] + 0.5*dy[i-1] + 0.5*dy[i]

        def get_boundary_time(x):
            return (x[:,-2] + x[:,-3])/2

        def get_boundary(x):
            return (x[-2] + x[-3])/2

        # ----- Recycling
        recycle_multiplier = float(o["d+"]["recycle_multiplier"])

        # ----- Boundary flux
        sheath_area = get_boundary(J) / np.sqrt(get_boundary(g_22))
        sheath_ion_flux = get_boundary_time(d["NVd+"].squeeze())
        sheath_ion_flux *= sheath_area * d["Cs0"] * d["Nnorm"] * mass_p / mass_i

        sheath_neutral_flux = get_boundary_time(d["NVd"].squeeze())
        sheath_neutral_flux *= sheath_area * d["Cs0"] * d["Nnorm"] * mass_p / mass_i

        intended_recycle_flux = sheath_ion_flux * recycle_multiplier

        # ----- Density input source
        if "Sd+_src" in d.keys():
            input_source = np.trapz(x = pos[2:-2], y = d["Sd+_src"].squeeze()[:,2:-2] * J[2:-2] * d["Cs0"] * d["Nnorm"])
        else:
            input_source = np.zeros_like(sheath_ion_flux)

        if "Sd_src" in d.keys():
            input_n_source = np.trapz(x = pos[2:-2], y = d["Sd_src"].squeeze()[:,2:-2] * J[2:-2] * d["Cs0"] * d["Nnorm"])
        else:
            input_n_source = np.zeros_like(sheath_ion_flux)

        # ----- Ionisation source
        if "Sd+_iz" in d.keys():
            iz_source = np.trapz(x = pos[2:-2], y = d["Sd+_iz"].squeeze()[:,2:-2] * J[2:-2] * d["Cs0"] * d["Nnorm"])
        else:
            iz_source = np.zeros_like(sheath_ion_flux)

        # ----- Recombination source
        if "Sd+_rec" in d.keys():
            rec_source = np.trapz(x = pos[2:-2], y = d["Sd+_rec"].squeeze()[:,2:-2] * J[2:-2] * d["Cs0"] * d["Nnorm"])
        else:
            rec_source = np.zeros_like(sheath_ion_flux)

        # ----- Total SNd+
        if "SNd+" in d.keys():
            alt_source = np.trapz(x = pos[2:-2], y = d["SNd+"].squeeze()[:,2:-2] * J[2:-2] * d["Cs0"] * d["Nnorm"])
        else:
            alt_source = np.zeros_like(sheath_ion_flux)

        # ----- Total ddt(Nd+)
        if "ddt(Nd+)" in d.keys():
            ddt_int = np.trapz(x = pos[2:-2], y = d["ddt(Nd+)"].squeeze()[:,2:-2] * J[2:-2] * d["Cs0"] * d["Nnorm"])
            ddt_last = d["ddt(Nd+)"].squeeze()[:,2:-2] * d["Cs0"] * d["Nnorm"]
        else:
            ddt_int = np.zeros_like(sheath_ion_flux)


        # ----- Totals
        total_ions = np.trapz(x = pos[2:-2], y = d["Nd+"].squeeze()[:,2:-2] * J[2:-2] * d["Nnorm"])
        total_neutrals = np.trapz(x = pos[2:-2], y = d["Nd"].squeeze()[:,2:-2] * J[2:-2] * d["Nnorm"])
        total_particles = total_ions + total_neutrals

        case_ion_source = np.trapz(x = pos[2:-2], y = d["SNd+"].squeeze()[:,2:-2] * J[2:-2] * d["Cs0"]  * d["Nnorm"])

        avg_dens = np.trapz(x = pos[2:-2], y = d["Nd+"].squeeze()[:,2:-2] * J[2:-2] * d["Nnorm"]) / sum(dy[2:-2])

        total_in = input_source + iz_source 
        total_out = sheath_ion_flux + abs(rec_source)
        total_balance = total_in - total_out
        frac_balance = total_balance / total_in

        neutral_in = rec_source
        neutral_out = iz_source

        fig, axes = plt.subplots(1,3, figsize=(18,4), dpi = 100)
        fig.suptitle(self.casename)
        fig.subplots_adjust(wspace=0.4)
        t = range(len(iz_source))


        ax = axes[0]
        ax.set_title("Domain particle sources/sinks")
        ax.plot(t, input_source, label = "Input plasma source")
        ax.plot(t, input_n_source, label = "Input neutral source", c = "k", zorder = 100)
        ax.plot(t, iz_source, label = "Ionisation source")
        ax.plot(t, rec_source, label = "Recombination sink")
        ax.plot(t, sheath_ion_flux, ls = "-", c = "grey", label = "Ion sheath sink")
        ax.set_ylabel("Particle flux [s-1]")


        ax = axes[1]
        ax.set_title("Total in/out, mass imbalance")
        ax.plot(t, total_in, lw = 2, ls = "-", c = "k", label = "Total in")
        ax.plot(t, total_out, lw = 2, ls = "-", c = "r", label = "Total out")
        ax.set_ylabel("Particle flux [s-1]")

        ax2 = ax.twinx()
        ax2.plot(t, frac_balance, lw = 2, alpha = 0.3, ls = "-", c = "magenta", label = "Imbalance")
        ax2.set_ylim(-1,1)
        ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.0%}"))
        ax2.set_ylabel("Mass imbalance [%]", c = "magenta")
        ax2.spines["right"].set_color("magenta")
        ax2.yaxis.label.set_color("magenta")
        ax2.tick_params(axis="y", colors = "magenta")


        ax = axes[2]
        ax.set_title("Total particle count, upstream density")
        ax.plot(t, total_particles, label = "Total domain particle count")
        ax.plot(t, total_ions, label = "Total domain ion count")
        ax.plot(t, total_neutrals, label = "Total domain neutral count")
        ax.set_ylabel("Particle count")

        ax2 = ax.twinx()
        ax2.plot(t, avg_dens, c = "r", ls = "-")
        ax2.set_ylabel("Upstream plasma density")
        ax2.spines["right"].set_color("red")
        ax2.yaxis.label.set_color("red")
        ax2.tick_params(axis="y", colors = "red")

        for ax in axes:
            ax.grid(which = "both")
            ax.set_xlabel("Timestep")


        axes[0].legend(loc="upper left", bbox_to_anchor = (-0.11,-0.15), ncol = 3)
        axes[1].legend(loc="upper left", bbox_to_anchor = (0.15,-0.15), ncol = 1)
        axes[2].legend(loc="upper left", bbox_to_anchor = (0.15,-0.15), ncol = 1)

        print(">>> System mass balance")
        print("- Total in ---------------")
        print(f"- Input ion source = {input_source[-1]:.3E} [s-1]")
        print(f"- Input neutral source = {input_n_source[-1]:.3E} [s-1]")
        print(f"- Ionisation source = {iz_source[-1]:.3E} [s-1]")
        print(f"- Intended recycling source = {iz_source[-1]:.3E} [s-1]")
        print(f"- Total = {total_in[-1]:.3E} [s-1]")
        print("\n- Total out ---------------")
        print(f"- Sheath ion flux = {sheath_ion_flux[-1]:.3E} [s-1]")
        print(f"- Sheath neutral flux = {sheath_neutral_flux[-1]:.3E} [s-1]")
        print(f"- Recombination source = {rec_source[-1]:.3E} [s-1]")
        print(f"- Total = {total_out[-1]:.3E} [s-1]")
        print(f"\n- Difference:")
        print(f"---> {total_balance[-1]:.3E} [s-1] ({total_balance[-1]/total_in[-1]:.3%})")



    def mass_balance(self,          
        verbosity = False, 
        plot = False, 
        output = False, 
        nloss_guardcells = True, 
        flux_input = False,
        NVi_is_NeVi = False,
        additional_plots = False,
        error_only = True):
        # TODO Potentially wrong as momentum should be Nnorm * Cs0
        """ 
        >>> WARNING this code needs to be redone. Need to do trapz(x=pos, y=source * J) to do volume integral!!
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
        d["Nnorm"] = collect("Nnorm", path = path_case)
        d["Cs0"] = collect("Cs0", path = path_case)
        d["Omega_ci"] = collect("Omega_ci", path = path_case)
        d["dy"] = collect("dy", path = path_case, yguards = True, info = False, strict = True)[0,1:-1]
        d["dy_full"] = collect("dy", path = path_case, yguards = True, info = False, strict = True)[0,:] 
        d["J_full"] = collect("J", path = path_case, yguards = True, info = False, strict = True)[0,:] 
        

        if self.hermes:
            h["Vi"] = collect("Vd+", path = path_case, yguards = True, info = False, strict = True)[:,0,1:-1,0]
            h["NVi_original"] = collect("NVd+", path = path_case, yguards = True, info = False, strict = True)[:,0,1:-1,0]
            nloss = 0 # Can't find nloss in hermes-3
            frecycle_intended = self.options["d+"]["recycle_multiplier"]
            if "upstream_density_feedback" in self.options.keys():
                Nu_target = self.options["upstream_density_feedback"]["density_upstream"]
            else:
                Nu_target = np.nan
        else:
            h["Vi"] = collect("Vi", path = path_case, yguards = True, info = False, strict = True)[:,0,1:-1,0]
            h["NVi_original"] = collect("NVi", path = path_case, yguards = True, info = False, strict = True)[:,0,1:-1,0]
            nloss = self.options["sd1d"]["nloss"]
            frecycle_intended = self.options["sd1d"]["frecycle"]
            if "density_upstream" in self.options["sd1d"].keys():
                Nu_target = self.options["sd1d"]["density_upstream"]
            else:
                Nu_target = np.nan

        if flux_input == True:
            flux_intended = self.options["Ne"]["flux"]
        else:
            flux_intended = 0


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


        #>>>>>>> Save data to class
        self.flux_sheath = flux_sheath[-1]
        self.flux_sink = flux_sink[-1]
        self.flux_source = flux_source[-1]
        self.flux_nloss = flux_nloss[-1]
        self.flux_in = flux_in[-1]
        self.flux_intended = flux_intended
        self.evolving_source = evolving_source
        self.frecycle = frecycle[-1]
        self.frecycle_intended = frecycle_intended
        self.frecycle_error = frecycle_error[-1]
        self.flux_imbalance = flux_imbalance[-1]

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

    

    def sheath_boundary_simple(case):
        """
        Replication of sheath_boundary_simple.cxx
        Done for the upper_y boundary only for clarity
        Works only for a single ion and single neutral species atm!
        WIP
        """

        casepath = casepaths["hrac-c5-10-fixbc"]
        d = BoutData(casepath, yguards = True, info = False, strict = True, DataFileCaching=False)["outputs"]
        o = BoutData(casepath, yguards = True, info = False, strict = True, DataFileCaching=False)["options"]

        # ----- Options functions and constanats
        gamma_e = float(o["sheath_boundary_simple"]["gamma_e"])
        gamma_i = float(o["sheath_boundary_simple"]["gamma_i"])

        try:
            Ge = float(o["sheath_boundary_simple"]["secondary_electron_coef"])
        except:
            Ge = 0 # Default
            pass

        try:
            sheath_ion_polytropic = float(o["sheath_ion_polytropic"])
        except:
            sheath_ion_polytropic = 1 # Default
            pass

        Zi = 1

        def get_boundary(x):
            return (x[-2] + x[-3])/2

        def extract(x):
            return x.squeeze()[tind,:]

        def limitFree(fm, fc):
            #  fm  fc | fp
            #         ^ boundary
            # Linear extrapolation & limit
            # To force to 0 gradient if var increasing into sheath
            if fm < fc:
                return fc
            if fm < 1e-10:
                return fc
            
            fp = fc**2 / fm
            return fp

        tind = -1

        mass_i = constants("mass_p") * 2 # [kg]
        mass_e = constants("mass_e") # [kg]
        q_e = constants("q_e") # [J/eV]
        k_b = constants("k_b") # [J/K]

        Vnorm = d["Cs0"]
        Enorm = q_e * d["Nnorm"] * d["Tnorm"] * d["Omega_ci"] # [Wm-3]
        Xnorm = d["Cs0"] * d["Nnorm"] # [m-2s-1]
        Nnorm = d["Nnorm"] # [m-3]
        Tnorm = d["Tnorm"] # [eV]
        Bnorm = d["Bnorm"] # [T]
        Pnorm = Nnorm * Tnorm * q_e # [Pa]
        Fnorm = mass_i * Nnorm * d["Cs0"] * d["Omega_ci"]

        """
        Hey @mikekryjak It's the momentum density, rather than the particle flux. Dividing by AA gives the particle flux as it was in SD1D.
        For example, the NVd+ output if multiplied by m_p * Nnorm * Cs0 is the momentum density, or equivalently mass flux, with units of [kg m^-2 s^-1]. m_p here is the proton mass.
        Taking out the AA factor gives particle flux, so NVd+ / 2 multiplied by Nnorm * Cs0 is the particle flux with units of [m^-2 s^-1]
        """

        # ----- Unpack and denormalise things
        Zi = 1
        AA = 2
        NVi = extract(d["NVd+"]) * Xnorm * mass_p # Momentum density [kgm-2s-1]
        Ve = extract(d["Ve"]) * Vnorm
        Vi = extract(d["Vd+"]) * Vnorm
        Ti = extract(d["Td+"]) * Tnorm
        Te = extract(d["Te"]) * Tnorm
        Ni = extract(d["Nd+"]) * Nnorm
        Ne = extract(d["Ne"]) * Nnorm
        Pe = extract(d["Ne"]) * Pnorm
        Pi = extract(d["Pd+"]) * Pnorm
        i_hflux_add = -1 * d["Ed+_sheath"].squeeze()[-1,:][-3] * dV[-3] * Enorm * 1e-6
        e_hflux_add = -1 * d["Ee_sheath"].squeeze()[-1,:][-3] * dV[-3] * Enorm * 1e-6
        case_J = d["J_sheath"].squeeze()[-1,:] * Nnorm * d["Cs0"]
        case_phi = d["phi_sheath"].squeeze()[-1,:] * Tnorm


        # ----- Geometry
        J = d["J"].squeeze() * d["rho_s0"] / Bnorm # from hermes-3.cxx
        g_22 = d["g_22"].squeeze() * d["rho_s0"]**2 # from hermes-3.cxx
        dy = d["dy"].squeeze()
        dV = J * dy
        A = get_boundary(J) / np.sqrt(get_boundary(g_22)) # sheath area

        # ----- Calculate sheath and guard cell values
        # NOTATION to match upper_y
        # i = last cell index
        # ip = guard cell index
        # im = second to last cell index
        # Note we include all guard cells here (two on each end)
        im = -4
        i = -3

        Ne_ip = limitFree(Ne[im], Ne[i])
        Ni_ip = limitFree(Ni[im], Ni[i])
        Ti_ip = limitFree(Ti[im], Ti[i])

        Te_ip = limitFree(Te[im], Te[i])
        Pi_ip = limitFree(Pi[im], Pi[i])

        nesheath = 0.5 * (Ne_ip + Ne[i])
        nisheath = 0.5 * (Ni_ip + Ni[i])
        tesheath = 0.5 * (Te_ip + Te[i])
        tisheath = 0.5 * (Ti_ip + Ti[i])

        print(f"Ratio of calculated to model sheath values:")
        print(f"- Ne: x{nesheath/get_boundary(Ne)}")
        print(f"- Ni: x{nisheath/get_boundary(Ni)}")
        print(f"- Te: x{tesheath/get_boundary(Te)}")
        print(f"- Ti: x{tisheath/get_boundary(Ti)}")

        # ----- Ion flux

        C_i_sq = (sheath_ion_polytropic * tisheath*q_e + Zi * tesheath*q_e) / mass_i
        Cs = np.sqrt(C_i_sq)
        visheath = Cs
        nvisheath = mass_i*nisheath*visheath

        #  last | guard
        #       ^ sheath
        # sheath = (last + guard) / 2
        # -> guard = 2 * sheath - last
        Vi_ip = 2 * visheath - Vi[i]
        NVi_ip = 2 * nvisheath - NVi[i]

        print(f"- Vi: x{visheath/get_boundary(Vi):.1f} which is {visheath:.2E} [m/s]")
        print("")
        print(f"- Ratio of NVi/Ni/Vi/mass_p at guard = {NVi_ip / Vi_ip / Ne_ip / mass_p}")
        print(f"- Ratio of NVi/Ni/Vi/mass_p at sheath = {nvisheath / visheath / nesheath / mass_p}")
        print(f"- Note that the ratio is satisfied at the sheath, and the guard cell diverges due to extrapolation.")
        print("")

        # ----- Heat flux
        # Stangeby heat flux is 2.5*TH + 1.0*KE = 3.5xTH which is a gamma of 3.5 by definition
        # Which is calculated by solving pressure and momentum in the main equations
        # The sheath BC allows you to have a different gamma by adding or taking away heat 
        # Based on difference from chosen gamma to 3.5. 
        q_i = (gamma_i-2.5) * tisheath*q_e - 0.5*C_i_sq*mass_i * nisheath*visheath * 1e-6

        print(f"- Additional ion cooling in BC: {q_i:.2f} [MW] Model check: x{q_i / i_hflux_add:.2f}")

        # ----- Electron flux

        ion_sum = np.zeros_like(Ne)
        phi = np.zeros_like(Ne)

        ion_sum[i] = Zi * nisheath * Cs # Ion current
        phi[i] = tesheath * np.log(np.sqrt(tesheath*q_e / (mass_e * np.pi*2)) * (1-Ge) * nesheath / ion_sum[i])

        print(f"- Current in last cell: x{ion_sum[i]/case_J[-3]:.1f} which is {ion_sum[i]:.2E} [A]")
        print(f"- Potential in last cell: x{phi[i]/case_phi[-3]:.1f} which is {phi[i]:.2E} [A]")

        Ne_ip = limitFree(Ne[im], Ne[i])
        Te_ip = limitFree(Te[im], Te[i])
        Pe_ip = limitFree(Pe[im], Pe[i])

        # Phi not limited because it naturally increases into sheath!
        phi_ip = 2 * phi[i] - phi[im]

        nesheath = 0.5 * (Ne_ip + Ne[i])
        tesheath = 0.5 * (Te_ip + Te[i])
        phisheath = np.clip(0.5 * (phi_ip + phi[i]), 0, None) # Electron saturation at phi = 0

        vesheath  = np.sqrt((tesheath)*q_e / (2*np.pi*mass_e) ) * (1-Ge) * np.exp(-phisheath / tesheath)
        # vesheath = np.sqrt(tesheath)*q_e * (np.sqrt(mass_e) / (2 * np.sqrt(np.pi))) * np.exp(-phisheath / tesheath)

        #  sqrt(tesheath) * (sqrt(mi_me) / (2. * sqrt(PI))) * exp(-phi_te);
        nvesheath = mass_e*nesheath*vesheath

        Ve_ip = 2 * vesheath - Ve[i]
        NVe_ip = 2 * nvesheath - NVe[i]

        # Ve is supposed to be the same as Vi all the time, apart from when there are currents. It also is this in the code.
        # It is not so in my reproduction. This must be an error.

        print(f"- Ve: x{vesheath/get_boundary(Ve):.1f} which is {vesheath:.2E} [m/s]")


    def animate(self, param):
        ds = xbout.open_boutdataset(datapath = self.datapath, inputfilepath = self.inputfilepath,
                                    info = False, keep_yboundaries = True)
        ds = ds.squeeze(drop = True)
        xbout.plotting.animate.animate_line(ds[param])

    def plot(self, vars = [["Te", "Ne", "Nn"], ["S", "R", "P"], ["NVi", "M", "F"]],
             trim = True, scale = "auto", xlims = (0,0)):
        lib = library()

        colors = mike_cmap(10)
        lw = 2

        for list_params in vars:

            fig, axes = plt.subplots(1,3, figsize = (18,5))
            fig.subplots_adjust(wspace=0.4)

            for i, ax in enumerate(axes):
                param = list_params[i]

                data = self.data
                if param in data.keys():
                    ax.plot(data["pos"], data[param], color = colors[i], linewidth = lw)


                ax.set_xlabel("Position (m)")
                ax.set_ylabel("{} ({})".format(lib[param]["name"], lib[param]["unit"]), fontsize = 11)
                
                ax.set_yscale("log")
                if scale == "auto":
                    ax.set_yscale(lib[param]["scale"])
                elif scale == "log":
                    ax.set_yscale("log")
                elif scale == "symlog":
                    ax.set_yscale("symlog")
                    
                ax.set_title(param)
                # ax.legend(fontsize = 10)
                ax.grid(which="major", alpha = 0.3)
                zoomlims = (max(data["pos"])*0.91, max(data["pos"])*1.005)
                if trim and param in ["NVi", "P", "M", "Ne", "Nn",  "Fcx", 
                "Frec", "E", "F", "R", "Rex", "Rrec", "Riz", 
                "Siz", "S", "Eiz", "Vi", "Pn", "NVn", "Srec"]:
                    ax.set_xlim(zoomlims)

                if xlims != (0,0):
                    ax.set_xlim(xlims)

                ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1e}"))


    def get_timestats(self):

            self.wtime = self.raw_data["wtime"]
            self.wtime_sum = sum(self.raw_data["wtime"])/3600
            self.wtime_avg = np.mean(self.wtime)
            self.wtime_std = np.std(self.wtime)
            self.rhscalls = self.raw_data["wtime_rhs"]
            self.rhscalls_sum = sum(self.raw_data["wtime_rhs"])
            
    def check_iz(self):
        rtools = AMJUEL()
        pos = self.data["pos"]
        Te = self.data["Te"]
        Ne = self.data["Ne"]
        Nn = self.data["Nn"]
        Vi = self.data["Vi"]

        if self.evolve_pn:
            Tn = self.data["Tn"]
        if self.evolve_nvn:
            Vn = self.data["Vn"]
        else:
            Vn = np.zeros_like(Vi)
        if self.hermes:
            Ti = self.data["Ti"]
        else:
            Ti = Te

        # ----- Ionisation
        Siz = abs(self.data["Siz"])
        Eiz = self.data["Eiz"] * 1e-6 / (3/2)
        Fiz = self.data["Fiz"]

        freq_amj = np.zeros_like(Te)
        Siz_amj = np.zeros_like(Te)
        Eiz_amj = np.zeros_like(Te)
        Fiz_amj = np.zeros_like(Te)

        for i, _ in enumerate(pos):
            freq_amj[i] = rtools.amjuel_2d("H.4 2.1.5", Te[i], Ne[i]) * Ne[i] * Nn[i]
            Siz_amj[i] = freq_amj[i]

            if self.evolve_pn:
                Eiz_amj[i] = freq_amj[i] * Tn[i] * constants("q_e") * 1e-6
            if self.evolve_nvn:
                Fiz_amj[i] = freq_amj[i] * Vn[i] * constants("mass_p") * 2


        fig, axes = plt.subplots(1,3, figsize = (15,5))
        fig.subplots_adjust(wspace = 0.4)

        ax = axes[0]
        ax.plot(pos, Siz, label = "Case", c = "k")
        ax.plot(pos, Siz_amj, label = "AMJ H.4 2.1.5", ls = ":", c = "r")
        ax.set_title("Ionisation")
        ax.set_ylabel("Ionisation rate [m-3s-1]")

        ax = axes[1]
        if self.evolve_pn:
            ax.plot(pos, Eiz, label = "case", c = "k")
            ax.plot(pos, Eiz_amj, label = "Calc", ls = ":", c = "r")
        ax.set_title("iz. energy transfer")
        ax.set_ylabel("Energy source [MWm-3]")

        ax = axes[2]
        if self.evolve_nvn:
            ax.plot(pos, Fiz, label = "case", c = "k")
            ax.plot(pos, Fiz_amj, label = "Calc", ls = ":", c = "r")
        ax.set_title("iz. mom transfer")
        ax.set_ylabel("Momentum source [Nm-3]")

        for ax in axes:
            ax.set_xlabel("pos [m]")
            ax.legend()
            ax.set_xlim(np.max(pos) * 0.995, np.max(pos) * 1.001)

    def check_rec(self):
        """
        Something is not right here in the energy and momentum
        Probably related to sheath ..it's the last few values
        """
       
        rtools = AMJUEL()
        pos = self.data["pos"]
        Te = self.data["Te"]
        Ne = self.data["Ne"]
        Nn = self.data["Nn"]
        Vi = self.data["Vi"]

        if self.evolve_pn:
            Tn = self.data["Tn"]
        if self.evolve_nvn:
            Vn = self.data["Vn"]
        else:
            Vn = np.zeros_like(Vi)
        if self.hermes:
            Ti = self.data["Ti"]
        else:
            Ti = Te

        # ----- Ionisation
        Srec = abs(self.data["Srec"])
        Erec = self.data["Erec"] * 1e-6 / (3/2) # TODO figure out why on earth the 3/2 is needed
        Frec = self.data["Frec"]

        freq_amj = np.zeros_like(Te)
        Srec_amj = np.zeros_like(Te)
        Erec_amj = np.zeros_like(Te)
        Frec_amj = np.zeros_like(Te)

        for i, _ in enumerate(pos):
            freq_amj[i] = rtools.amjuel_2d("H.4 2.1.8JH", Te[i], Ne[i]) * Ne[i]**2
            Srec_amj[i] = freq_amj[i]

            if self.evolve_pn:
                Erec_amj[i] = -freq_amj[i] * Ti[i] * constants("q_e") * 1e-6
            if self.evolve_nvn:
                # Frec_amj[i] = -freq_amj[i] * (Vi[i] - Vn[i]) * constants("mass_p") * 2
                Frec_amj[i] = -freq_amj[i] * Vi[i] * constants("mass_p") * 2 


        fig, axes = plt.subplots(1,3, figsize = (15,5))
        fig.subplots_adjust(wspace = 0.4)

        ax = axes[0]
        ax.plot(pos, Srec, label = "Case", c = "k")
        ax.plot(pos, Srec_amj, label = "AMJ H.4 2.1.8", ls = ":", c = "r")
        ax.set_title("Recombination freq")
        ax.set_ylabel("Recombination rate [m-3s-1]")

        ax = axes[1]
        if self.evolve_pn:
            ax.plot(pos, Erec, label = "case", c = "k")
            ax.plot(pos, Erec_amj, label = "AMJ", ls = ":", c = "r")
        ax.set_title("Rec. energy transfer")
        ax.set_ylabel("Energy source [MWm-3]")

        ax = axes[2]
        if self.evolve_nvn:
            ax.plot(pos, Frec, label = "case", c = "k")
            ax.plot(pos, Frec_amj, label = "AMJ", ls = ":", c = "r")
        ax.set_title("Rec. mom transfer")
        ax.set_ylabel("Momentum source [Nm-3]")

        for ax in axes:
            ax.set_xlabel("pos [m]")
            ax.legend()
            ax.set_xlim(np.max(pos) * 0.995, np.max(pos) * 1.001) 



    def check_rad(self):
        rtools = AMJUEL()
        pos = self.data["pos"]
        Te = self.data["Te"]
        Ne = self.data["Ne"]
        Nn = self.data["Nn"]

        # ----- Ionisation
        Rex = abs(self.data["Rex"]) * 1e-6
        Rrec = abs(self.data["Rrec"]) * 1e-6

        Rex_amj = np.zeros_like(Te)
        Rrec_amj = np.zeros_like(Te)
        Rrec_base = np.zeros_like(Te)
        Rrec_rad = np.zeros_like(Te)

        for i, _ in enumerate(pos):
            Rex_amj[i] = rtools.amjuel_2d("H.10 2.1.5", Te[i], Ne[i]) * Ne[i] * Nn[i] * constants("q_e") * 1e-6 #MW

            # The ion releases 13.6eV because it gets recombined (so 13.6eV of heating)
            # The resulting neutral de-excites and releases a photon equivalent to total ion energy - 13.6eV
            # TODO understand this as it's likely >>> wrong
            Rrec_amj[i] = (-rtools.amjuel_2d("H.10 2.1.8", Te[i], Ne[i]) * Ne[i] * Ne[i] + \
                            rtools.amjuel_2d("H.4 2.1.8", Te[i], Ne[i]) * Ne[i] * Ne[i] * 13.6) * constants("q_e") * 1e-6

            Rrec_base[i] = rtools.amjuel_2d("H.4 2.1.8", Te[i], Ne[i]) * Ne[i] * Ne[i] * 13.6 * constants("q_e") * 1e-6
            Rrec_rad[i] = rtools.amjuel_2d("H.10 2.1.8", Te[i], Ne[i]) * Ne[i] * Ne[i] * constants("q_e") * 1e-6

        fig, axes = plt.subplots(1,3, figsize = (16,5))
        fig.subplots_adjust(wspace = 0.5)

        ax = axes[0]
        ax.plot(pos, Rex, label = "Case", c = "k")
        ax.plot(pos, Rex_amj, label = "AMJ H.10 2.1.5", ls = ":", c = "r")
        ax.set_xlim(np.max(pos) * 0.9, np.max(pos) * 1.01)
        ax.set_title("Excitation radiation")
        ax.set_ylabel("Radiation [Wm-3]")
        ax.set_xlabel("pos [m]")
        ax.legend()

        ax = axes[1]
        ax.plot(pos, Rrec, label = "Case", c = "k")
        ax.plot(pos, Rrec_amj, label = "AMJ H.10 2.1.8", ls = ":", c = "r")
        ax.set_xlim(np.max(pos) * 0.99, np.max(pos) * 1.001)
        ax.set_title("Recombination radiation")
        ax.set_ylabel("Radiation [Wm-3]")
        ax.set_xlabel("pos [m]")
        ax.legend()

        ax = axes[2]
        ax.plot(pos, Rrec_base, label = "13.6 cost", c = "navy")
        ax.plot(pos, Rrec_rad, label = "Total radiation",  c = "darkorange")
        ax.set_xlim(np.max(pos) * 0.99, np.max(pos) * 1.001)
        ax.set_title("Recombination radiation")
        ax.set_ylabel("Radiation [Wm-3]")
        ax.set_xlabel("pos [m]")
        ax.legend()

    def check_cx(self):
        # ----- Charge exchange
        rtools = AMJUEL()
        atools = Atomics()
        pos = self.data["pos"]
        Te = self.data["Te"]
        Ne = self.data["Ne"]
        Nn = self.data["Nn"]
        Vi = self.data["Vi"]
        mass_i = constants("mass_p") * 2
        Fcx = self.data["Fcx"]
        Ecx = self.data["Ecx"]

        Ti = self.data["Ti"] if self.ion_eqn else Te
        Tn = self.data["Tn"] if self.evolve_pn else Ti
        Vn = self.data["Vn"] if self.evolve_nvn else 0

        if self.hermes:
            sigmav_amj = np.zeros_like(Te) # CX frequency
            print("This is a Hermes case. Comparing to H.2 3.1.8 low E limit AMJUEL rate.")
            for i, _ in enumerate(pos):
                sigmav_amj[i] = rtools.amjuel_1d("H.2 3.1.8", Te[i])
        else:
            sigmav_amj = atools.cx_willett(Te)
            print("This is an SD1D case. Comparing to H.3 3.1.8 @ E=10eV.")

        rate_amj = sigmav_amj * Ne * Nn

        ion_mom = rate_amj * Vi * mass_i
        atom_mom = rate_amj * Vn * mass_i
        Fcx_amj = ion_mom - atom_mom

        ion_energy = (3/2) * rate_amj * Ti * constants("q_e")
        atom_energy = (3/2) * rate_amj * Tn * constants("q_e")
        Ecx_amj = ion_energy - atom_energy

        # ----- Plot
        fig, axes = plt.subplots(1,2, figsize = (10,5))
        fig.subplots_adjust(wspace=0.3)

        ax = axes[0]
        ax.plot(pos[:-1], Fcx[:-1], label = "Fcx (case)", c = "k")
        ax.plot(pos[:-1], Fcx_amj[:-1], label = "Fcx (amjuel)", ls = ":", c = "r")
        ax.set_title("Ion momentum sink")
        ax.set_ylabel("Momentum [Nm-3]")

        ax = axes[1]
        ax.plot(pos[:-1], Ecx[:-1], label = "Ecx (case)", c = "k")
        ax.plot(pos[:-1], Ecx_amj[:-1], label = "Ecx (amjuel)", ls = ":", c = "r")
        ax.set_title("Ion energy sink")
        ax.set_ylabel("Power [Wm-3]")

        [ax.set_xlim(np.max(pos) * 0.99, np.max(pos) * 1.002) for ax in axes]
        [ax.legend() for ax in axes]
        [ax.set_xlabel("Pos [m]") for ax in axes]

    def check_dn(self):
        # ----- Charge exchange
        rtools = AMJUEL()
        atools = Atomics()
        fION = atools.iz_willett # SD1D iz rate
        pos = self.data["pos"]
        Te = self.data["Te"]
        Ne = self.data["Ne"]
        Nn = self.data["Nn"]
        Dn = self.data["Dn"]
        
        mass_i = constants("mass_p") * 2
        q_e = constants("q_e")

        Ti = self.data["Ti"] if self.ion_eqn else Te
        Tn = self.data["Tn"] if self.evolve_pn else Ti

        # ----- Calculate CX rate
        if self.hermes:
            sigmav_amj = np.zeros_like(Te) # CX frequency
            print("This is a Hermes case. Comparing to H.2 3.1.8 low E limit AMJUEL rate.")
            for i, _ in enumerate(pos):
                sigmav_amj[i] = rtools.amjuel_1d("H.2 3.1.8", Te[i])
        else:
            sigmav_amj = atools.cx_willett(Te)
            print("This is an SD1D case. Comparing to H.3 3.1.8 @ E=10eV.")

        nu_cx = sigmav_amj * Ne
        
        # ----- NN rate (SD1D only):
        vth_n = np.sqrt(Tn*q_e/mass_i) 
        a0 = np.pi * 5.29e-11**2
        lambda_nn = 1/(Nn*a0)
        for i, _ in enumerate(lambda_nn):
            if lambda_nn[i] > 0.1:
                lambda_nn[i] = 0.1
        nu_nn = vth_n / lambda_nn

        # ----- IZ rate (SD1D only):
        nu_iz = Ne * fION(Te)

        # ----- Assemble Dn 
        if self.hermes:
            Dn_calc = self.dneut * vth_n**2 / (nu_cx)
        else:
            Dn_calc = self.dneut * vth_n**2 / (nu_cx + nu_iz + nu_nn)

        # ----- Plot
        zoom = 0.8
        zoom_idx = int(len(pos)*zoom)
        fig, ax = plt.subplots(figsize = (6,5))
        fig.subplots_adjust(wspace=0.3)
        ax.plot(pos[zoom_idx:-1], Dn[zoom_idx:-1], label = "Dn (case)", c = "k")
        ax.plot(pos[zoom_idx:-1], Dn_calc[zoom_idx:-1], label = "Dn (check)", ls = ":", c = "r")
        ax.set_title("Neutral diffusion coefficient")
        ax.set_xlabel("Pos [m]")
        ax.set_ylabel("Dn [m-2s-1]")
        ax.set_xlim(pos[zoom_idx], np.max(pos) * 1.002)
        ax.legend()


class CaseDeck:
    def __init__(self, path, key = "", keys = [], skip = [], explicit = [], verbose = False):

        self.casepaths_all = dict()
        self.casepaths = dict()
        self.cases = dict()
        for root, dirs, files in os.walk(path):
            for file in files:
                if ".dmp" in file:
                    case = os.path.split(root)[1]
                    self.casepaths_all[case] = root

                    if explicit != []:
                        if key in root:
                            self.casepaths[case] = root

                    else:

                        if key != "" and any(x not in case for x in skip) == False:
                            if key in root:
                                self.casepaths[case] = root

                        elif keys == [] and any(x not in case for x in skip) == False:
                            self.casepaths[case] = root

                        if keys != []:
                            if any(x in case for x in keys) and any(x in case for x in skip) == False:
                                self.casepaths[case] = root

        self.casenames_all = list(self.casepaths_all.keys())
        self.casenames = list(self.casepaths.keys())

        if verbose:
            print("\n>>> All cases in path:", self.casenames_all)
            print(f"\n>>> All cases matching the key '{key}': {self.casenames}\n")
        
        print(f">>> Loading cases: ", end="")

        self.suffix = dict()
        for case in self.casenames:
            print(f"{case}... ", end="")
            self.cases[case] = Case(self.casepaths[case])
            
            suffix = case.split("-")[-1]
            self.suffix[suffix] = self.cases[case]
            
            self.cases[case].get_htransfer()


        self.get_stats()
        
        print("...Done")

    def get_stats(self):
        self.stats = pd.DataFrame()
        self.stats.index.rename("case", inplace = True)

        for casename in self.casenames:
            case = self.cases[casename]
            self.stats.loc[casename, "target_flux"] = case.data["NVi"][-1]
            self.stats.loc[casename, "target_temp"] = case.data["Te"][-1]

            if case.hermes:
                self.stats.loc[casename, "initial_dens"] = case.options["Nd+"]["function"] * case.Nnorm
                self.stats.loc[casename, "line_dens"] = case.options["Nd+"]["function"] * case.Nnorm
            else:
                self.stats.loc[casename, "initial_dens"] = case.options["Ne"]["function"] * case.Nnorm
                self.stats.loc[casename, "line_dens"] = case.options["Ne"]["function"] * case.Nnorm

        self.stats.sort_values(by="initial_dens", inplace=True)

    def get_heat_balance(self):
        self.heat_balance = pd.DataFrame()
        
        for casename in self.casenames:
            case = self.cases[casename]
            case.heat_balance()
            self.heat_balance.loc[casename, "hflux_E"] = case.hflux_E
            self.heat_balance.loc[casename, "hflux_R"] = case.hflux_R
            self.heat_balance.loc[casename, "hflux_F"] = case.hflux_F
            self.heat_balance.loc[casename, "hflux_sheath"] = case.hflux_sheath
            self.heat_balance.loc[casename, "All out"] = case.hflux_out
            self.heat_balance.loc[casename, "hflux_source"] = case.hflux_source
            self.heat_balance.loc[casename, "hflux_imbalance"] = case.hflux_imbalance
            self.heat_balance.loc[casename, "hflux_imbalance_ratio"] = case.hflux_imbalance / case.hflux_source

            if case.snb:
                self.heat_balance.loc[casename, "sheath SNB"] = case.hflux_lastcell_snb
                self.heat_balance.loc[casename, "sheath SH"] = case.hflux_lastcell_sh

        print("Heat flows in MW:")
        display(self.heat_balance)

    def plot(self, vars = [["Te", "Ne", "Nn"], ["S", "R", "P"], ["NVi", "M", "F"]],
             trim = True, scale = "auto", xlims = (0,0)):
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
                    if param in data.keys():
                        ax.plot(data["pos"], data[param], color = colors[i], linewidth = lw, label = case)


                ax.set_xlabel("Position (m)")
                ax.set_ylabel("{} ({})".format(lib[param]["name"], lib[param]["unit"]), fontsize = 11)
                
                ax.set_yscale("log")
                if scale == "auto":
                    ax.set_yscale(lib[param]["scale"])
                elif scale == "log":
                    ax.set_yscale("log")
                elif scale == "symlog":
                    ax.set_yscale("symlog")
                    
                ax.set_title(param)
                ax.legend(fontsize = 10)
                ax.grid(which="major", alpha = 0.3)
                zoomlims = (max(data["pos"])*0.91, max(data["pos"])*1.005)
                if trim and param in ["NVi", "P", "M", "Ne", "Nn",  "Fcx", 
                "Frec", "E", "F", "R", "Rex", "Rrec", "Riz", 
                "Siz", "S", "Eiz", "Vi", "Pn", "NVn"]:
                    ax.set_xlim(zoomlims)

                if xlims != (0,0):
                    ax.set_xlim(xlims)

                ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1e}"))

class Atomics():
    def __init__(self):
        pass

    def iz_willett(self, Te, coeffs = [-3.271397E1, 1.353656E1, -5.739329, 1.563155, -2.877056E-1, 3.482560e-2, -2.631976E-3, 1.119544E-4, -2.039150E-6]):
        """ SD1D ionisation rate code"""
        
        TT = Te
        lograte = 0.0
        
        for i, coeff in enumerate(coeffs):
            lograte = lograte +  coeff * np.log(TT)**(i)
        fION = np.exp(lograte)*1e-6

        return fION

    def ex_willett(self, Te):
        """ SD1D excitation energy rate code"""

        TT = Te
        Y = 10.2 / TT
        fEXC = 49.0E-14/(0.28+Y)*np.exp(-Y)*np.sqrt(Y*(1.0+Y))
    
        return fEXC

    def cx_willett(self, Te, E = 10):
        """ SD1D charge exchange rate code"""

        cxcoeffs = [
        [
            -1.829079582E1, 1.640252721E-1, 3.364564509E-2, 9.530225559E-3,
            -8.519413900E-4, -1.247583861E-3, 3.014307546E-4, -2.499323170E-5,
            6.932627238E-7,
        ],
        [
            2.169137616E-1, -1.106722014E-1, -1.382158680E-3, 7.348786287E-3,
            -6.343059502E-4, -1.919569450E-4, 4.075019352E-5, -2.850044983E-6,
            6.966822400E-8,
        ],
        [
            4.307131244E-2, 8.948693625E-3, -1.209480567E-2, -3.675019470E-4,
            1.039643391E-3, -1.553840718E-4, 2.670827249E-6, 7.695300598E-7,
            -3.783302282E-8,
        ],
        [
            -5.754895093E-4, 6.062141761E-3, 1.075907882E-3, -8.119301728E-4,
            8.911036876E-6, 3.175388950E-5, -4.515123642E-6, 2.187439284E-7,
            -2.911233952E-9,
        ],
        [
            -1.552077120E-3, -1.210431588E-3, 8.297212634E-4, 1.361661817E-4,
            -1.008928628E-4, 1.080693990E-5, 5.106059414E-7, -1.299275586E-7,
            5.117133050E-9,
        ],
        [
            -1.876800283E-4, -4.052878752E-5, -1.907025663E-4, 1.141663042E-5,
            1.775681984E-5, -3.149286924E-6, 3.105491555E-8, 2.274394089E-8,
            -1.130988251E-9,
        ],
        [
            1.125490271E-4, 2.875900436E-5, 1.338839629E-5, -4.340802793E-6,
            -7.003521917E-7, 2.318308730E-7, -6.030983538E-9, -1.755944926E-9,
            1.005189187E-10,
        ],
        [
            -1.238982763E-5, -2.616998140E-6, -1.171762874E-7, 3.517971869E-7,
            -4.928692833E-8, 1.756388999E-10, -1.446756796E-10, 7.143183138E-11,
            -3.989884106E-12,
        ],
        [
            4.163596197E-7, 7.558092849E-8, -1.328404104E-8, -9.170850254E-9,
            3.208853884E-9, -3.952740759E-10, 2.739558476E-11, -1.693040209E-12,
            6.388219930E-14,
        ],
    ]

        lograte = 0.0;
        for i in range(9):
            for j in range(9):
                lograte = lograte + cxcoeffs[i][j] * np.log(Te)**i * np.log(E)**j
        
        return 1.0E-6 * np.exp(lograte)

    def dn_sd1d(self, Te, Ne, Nn):
        
        """SD1D neutral diffusion rate Dn"""
   
        mass_p = 1.6726219e-27
        mass_i = mass_p * 2
        q_e = 1.60217662E-19

        fCX = self.cx_willett
        fION = self.iz_willett

        # Thermal velocity
        vth_n = np.sqrt(Te*q_e/mass_i) 

        # CX rate
        sigma_cx = Ne * fCX(Te)

        # IZ rate
        sigma_iz = Ne * fION(Te)

        # NN mfp - neutral-neutral elastic collisioms
        a0 = np.pi * 5.29e-11**2
        lambda_nn = 1/(Nn*a0)
        for i, _ in enumerate(lambda_nn):
            if lambda_nn[i] > 0.1:
                lambda_nn[i] = 0.1

        # NN rate
        sigma_nn = vth_n / lambda_nn

        # Neutral diffusion
        dn = vth_n**2 / (sigma_cx + sigma_iz + sigma_nn)
            
        return dn

    def pop_ex(self, Te, Ne):
        """
        Population method excitation
        Use H12 excited state population rates from AMJUEL
        And Einstein coefficients & bandgaps
        Originally written by Yulin Zhou
        """

        E = dict(); A = dict()
        E["21"] = 10.2;       E["31"] = 12.1;       E["41"] = 12.8;       E["51"] = 13.05;      E["61"] = 13.22
        A["21"] = 4.6986e8;   A["31"] = 5.5751e7;   A["41"] = 1.2785e7;   A["51"] = 4.1250e6;   A["61"] = 1.6440e6

        rates = ["H.12 2.1.5b", "H.12 2.1.5a", "H.12 2.1.5c", "H.12 2.1.5d", "H.12 2.1.5e"]
        einsteins = [A["21"], A["31"], A["41"], A["51"], A["61"]]
        bandgaps = [E["21"], E["31"], E["41"], E["51"], E["61"]]
        
        A21_check = 6.265e8
        A21_literature = 4.6986e8
        A21_code = 1.6986e9
        
        Rex = 0; R = dict()
        for i, n in enumerate(range(2,7)):
            R[n] = amjuel_2d([Te], Ne, amjuel_data[rates[i]], pop_ratio=True) * einsteins[i] * bandgaps[i]
            R[n] = R[n] / Ne  # eV/m3s to W/m3
            Rex += R[n]
            
        # Rex += amjuel_2d([Te], Ne, amjuel_data["H.4 2.1.5"]) * 13.6
            
        return Rex

class AMJUEL():
    """
    Toolkit for reading/writing AMJUEL and HYDHEL rates
    Also contains lots of rates itself.
    """
    def __init__(self):
        self.get_amjuel_data()

    def read_amjuel_2d(self, path):
        """
        # Reads AMJUEL coefficient table
        # You must copypaste it from AMJUEL pdf into Excel and do data to columns
        # Align it with the top left cell and save as csv. 
        # rows = T index, cols = E/n/other index.
        # Don't worry if you display the dataframe and see zeros, this is
        # because it's not scientific notation by default.
        # Needs Pandas
        """
        rate = pd.read_csv(path, header = None)
        rate = pd.concat([rate.loc[2:10,:3].reset_index(), rate.loc[13:21,:3].reset_index(), rate.loc[24:32,:3].reset_index()], axis = 1, ignore_index = True)
        rate = rate.drop(columns = [0,1,5,6,10,11])
        rate.columns = range(9)
        
        for col in rate.columns:
            for row in rate.index:
                if type(rate.loc[row,col]) != np.float64:
                    # Some rates have e and get picked up. Others have D and
                    # get picked up as strings.
                    rate.loc[row,col] = rate.loc[row,col].replace("D", "e")
                    rate.loc[row,col] = float(rate.loc[row,col])
            
        return rate

    def read_amjuel_1d(self, path):
    # for 1D 9 coeff rates
        data = pd.read_csv(path, header = None)
        data = data[[1,3,5]]
        rate = pd.concat([data.loc[0,:], data.loc[1,:], data.loc[2,:]])
        rate.index = range(len(rate))
        return np.array(rate)

    def amjuel_2d(
        self, name, Te, E, mode = "default", second_param = "Ne", pop_ratio = False):
        """
        Read 2D AMJUEL coefficients and calculate rate
        - Inputs: reaction name, single value of T and E
            (E=density in m-3, or beam energy in eV depending on process used)
        - Outputs: Array of rates in m-3 corresponding to array of temperatures and single E provided
        - Modes: "default" is full 2D fit, "coronal" takes only lowest density fit (E=0)

        When the second parameter is density, we use a 1E-14 scaling factor.
        This is because AMJUEL uses scaled density n_hat = n * 1e-8 and
        uses cm-3. So to scale and convert to m-3 it's 1e-14.
        H.4 2.1.5 and H.10 2.1.5 ionisation and excitation energy rates use density.
        """

        data = self.amjuel_data[name]

        if second_param == "Ne":
            E = E * 1e-14 
            
        rate = 0

        if mode == "default":
            for E_index in data.columns:
                for T_index in data.index:
                    rate = rate + \
                    data.loc[T_index,E_index] * \
                    (np.log(E)**E_index) * (np.log(Te)**T_index)
                    
        if mode == "coronal":
            E_index = 0
            for T_index in data.index:
                rate = rate + \
                data.loc[T_index,E_index] * \
                (np.log(E)**E_index) * (np.log(Te)**T_index)
                
        if pop_ratio:
            rate = np.exp(rate)
        else:
            rate = np.exp(rate)*1e-6 # cm-3 to m-3
    
        return rate

    def amjuel_1d(self, name, Te):
        """ 
        Read 1D AMJUEL coeff table and calc rate.
        No support for asymptotic parameters (al0, ar0 etc)
        TODO Potentially missing density correction of 1e14?
        """
        data = self.amjuel_data[name]

        rate = 0
        for i in range(len(data)):
            rate = rate + data[i] * np.log(Te)**i
            
        rate = np.exp(rate)*1e-6 # cm-3 to m-3
        
        return rate

    def print_rate(self, data, mode):
        """
        Prints AMJUEL rate as numpy array or C++ array
        for copy pasting
        Input is a read-in AMJUEL rate.
        Mode is Python or C++
        """

        if mode == "Python":

            print(r"np.array(")

            for i, col in enumerate(data.columns):
                if i == 0:
                    print("[", end = "")

                print("[")
                k = 0
                for i in range(3):
                    print("{:.12E}, {:.12E}, {:.12E},".format(data.loc[k,col], data.loc[k+1,col], data.loc[k+2,col]))
                    k += 3
                print("],", end = "")


            print("])")

        elif mode == "C++":
            for col in data.columns:
                print("      {")
                k = 0
                for i in range(3):
                    print("          {:.12E}, {:.12E}, {:.12E},".format(data.loc[k,col], data.loc[k+1,col], data.loc[k+2,col]))
                    k += 3
                print("      },")

    def get_amjuel_data(self):

        self.amjuel_data = dict()
        onedrive = r"C:\Users\mikek\OneDrive"

        # ALL FROM AMJUEL UNLESS OTHERWISE SPECIFIED
        # Ionisation particle source rate
        self.amjuel_data["H.4 2.1.5JH"] = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.4 2.1.5JH.csv")) # Ionisation, L.C.Johnson
        self.amjuel_data["H.4 2.1.5"] = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.4 2.1.5.csv")) # Ionisation, Sawada
        self.amjuel_data["H.4 2.1.5o"] = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.4 2.1.5o.csv")) # Ionisation, L.C.Johnson, Ly-opaque

        # Excitation energy weighted rate
        self.amjuel_data["H.10 2.1.5JH"] = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.10 2.1.5JH.csv")) # Excitation, L.C.Johnson
        self.amjuel_data["H.10 2.1.5"] = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.10 2.1.5.csv")) # Excitation, Sawada
        self.amjuel_data["H.10 2.1.5o"] = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.10 2.1.5o.csv")) # Excitation, L.C.Johnson, Ly-opaque
        self.amjuel_data["H.10 2.1.8"] = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.10 2.1.8.csv")) # Recombination, Sawada

        # Recombination particle source rate
        self.amjuel_data["H.4 2.1.8"]   = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.4 2.1.8.csv")) # Sawada, radiative + 3 body
        self.amjuel_data["H.4 2.1.8a"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.4 2.1.8a.csv")) # Johnson, radiative only
        self.amjuel_data["H.4 2.1.8b"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.4 2.1.8b.csv")) # Johnson, three body only
        self.amjuel_data["H.4 2.1.8JH"] = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.4 2.1.8JH.csv")) # Johnson, raditive + 3 body
        self.amjuel_data["H.4 2.1.8o"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.4 2.1.8o.csv")) # Johnson, Ly-opaque

        # Charge exchange
        self.amjuel_data["HYDHEL H.3 3.1.8"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\HYDHEL H.3 3.1.8.csv")) # Hydhel improved fit
        self.amjuel_data["H.3 3.1.8"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.3 3.1.8.csv")) # Same as hydhel equivalent
        self.amjuel_data["H.3 3.1.8org"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.3 3.1.8org.csv")) # Original fit before improvement
        self.amjuel_data["H.2 3.1.8"] = self.read_amjuel_1d(os.path.join(onedrive, r"Project\Atomicrates\1D H.2 3.1.8.csv")) # Low beam energy limit of 3.1.8. E=0
        self.amjuel_data["H.2 3.1.8FJ"] = self.read_amjuel_1d(os.path.join(onedrive, r"Project\Atomicrates\1D H.2 3.1.8FJ.csv")) # Variant of unknown origin, used in EDGE2D
        self.amjuel_data["H.2 0.1T"] = self.read_amjuel_1d(os.path.join(onedrive, r"Project\Atomicrates\1D H.2 0.1T.csv")) # Actually an elastic collision reaction - wrong.

        # Population of excited states of H
        self.amjuel_data["H.12 2.1.5"]   = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.12 2.1.5.csv")) # H+/H ratio, Reiter/Sawada/Fujimoto
        self.amjuel_data["H.12 2.1.5b"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.12 2.1.5b.csv")) # H(2)/H ratio, Reiter/Sawada/Fujimoto NOTE NUMBERING OUT OF ORDER
        self.amjuel_data["H.12 2.1.5a"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.12 2.1.5a.csv")) # H(3)/H ratio, Reiter/Sawada/Fujimoto NOTE NUMBERING OUT OF ORDER
        self.amjuel_data["H.12 2.1.5c"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.12 2.1.5c.csv")) # H(4)/H ratio, Reiter/Sawada/Fujimoto
        self.amjuel_data["H.12 2.1.5d"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.12 2.1.5d.csv")) # H(5)/H ratio, Reiter/Sawada/Fujimoto
        self.amjuel_data["H.12 2.1.5e"]  = self.read_amjuel_2d(os.path.join(onedrive, r"Project\Atomicrates\H.12 2.1.5e.csv")) # H(6)/H ratio, Reiter/Sawada/Fujimoto


def find_cases(path_studies):
    """ 
    Find all cases in all the studies in a studies folder
    Return dictionary with keys of case names and values of their paths
    """

    casepaths = dict()

    for folder in os.listdir(path_studies):

        path_study = path_studies + "\\" + folder

        for case in os.listdir(path_study):
            casepaths[case] = path_study + "\\" + case
            
    return casepaths

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
    lib["Pe"]["name"] = "Electron pressure"
    lib["Pe"]["unit"] = "Pa"
    lib["Te"]["name"] = "Plasma temperature"
    lib["Te"]["unit"] = "eV"
    lib["Ti"]["name"] = "Ion temperature"
    lib["Ti"]["unit"] = "eV"
    lib["Td+"]["name"] = "Ion temperature"
    lib["Td+"]["unit"] = "eV"
    lib["Td"]["name"] = "Deuterium temperature"
    lib["Td"]["unit"] = "eV"
    lib["Tn"]["name"] = "Neutral temperature"
    lib["Tn"]["unit"] = "eV"
    lib["Pn"]["name"] = "Neutral pressure"
    lib["Pn"]["unit"] = "Pa"
    lib["Pd"]["name"] = "Deuterium pressure"
    lib["Pd"]["unit"] = "Pa"
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

def heat_conduction(
    pos, Te, kappa0=2293.8117):
    """
    Calculate heat conduction in W/m^2 
    given input 1-D profiles
    
    pos[y] position in meters
    Te[y]  Electron temperature [eV]

    kappa0 Coefficient of heat conduction, so kappa = kappa0 * Te^5/2    

    Note: The value of kappa0 is what is used in SD1D
    """
    grad_Te = (Te[1:] - Te[:-1]) / (pos[1:] - pos[:-1]) # One-sided differencing

    Te_p = 0.5*(Te[1:] + Te[:-1])
    
    # Position of the result
    result_pos = 0.5*(pos[1:] + pos[:-1])
    
    return result_pos, -2293.8117*Te_p**(5./2)*grad_Te

def ke_convection(pos, n, vi, AA=2):
    """
    Calculate kinetic energy convection in W/m^2 
    given input 1-D profiles
    
    pos[y] position in meters
    n[y]   Density in m^-3
    vi[y]  Parallel flow velocity in m/s
    
    AA  Atomic mass number
    """

    # Interpolate onto cell boundaries
    vi_p = 0.5*(vi[1:] + vi[:-1])
    n_p = 0.5*(n[1:] + n[:-1])

    # Position of the result
    result_pos = 0.5*(pos[1:] + pos[:-1])

    return result_pos, 0.5*n_p*vi_p**3 * AA * 1.67e-27

def thermal_convection(
    pos, p, vi):
    """
    Calculate thermal energy convection in W/m^2 
    given input 1-D profiles
    
    pos[y] position in meters
    n[y]   Density in m^-3
    vi[y]  Parallel flow velocity in m/s
    """

    # Interpolate onto cell boundaries
    vi_p = 0.5*(vi[1:] + vi[:-1])
    p_p = 0.5*(p[1:] + p[:-1])

    # Position of the result
    result_pos = 0.5*(pos[1:] + pos[:-1])

    return result_pos, (5./2)*p_p*vi_p

def energy_flux(path, tind=-1):
    """
    Calculates the energy flux due to conduction and convection
    
    path  Path to the data files
    tind  Time index. By default the final time
    """
    verbosity = False
    
    # Evolving variables, remove extra guard cells so just one each side
    P = collect("P", path=path, tind=tind, yguards=True, info = verbosity)[-1,0,1:-1,0]
    Ne = collect("Ne", path=path, tind=tind, yguards=True, info = verbosity)[-1,0,1:-1,0]
    NVi = collect("NVi", path=path, tind=tind, yguards=True, info = verbosity)[-1,0,1:-1,0]

    # Normalisations
    nnorm = collect("Nnorm", path=path, tind=tind, info = verbosity)
    tnorm = collect("Tnorm", path=path, tind=tind, info = verbosity)
    pnorm = nnorm*tnorm*1.602e-19 # Converts p to Pascals
    cs0 = collect("Cs0", path=path)

    try:
        kappa_epar = collect("kappa_epar", path=path, tind=tind, info = verbosity)
    except:
        kappa_epar = None
    
    # electron temperature
    Te = (0.5*P/Ne) * tnorm

    # ion parallel velocity
    Vi = (NVi/Ne) * cs0

    NVi *= nnorm * cs0
    Ne *= nnorm
    P *= pnorm
    
    # Source
    pesource = collect("PeSource", path=path, yguards=True, info = verbosity)
    
    dy = collect("dy", path=path, yguards=True, info = verbosity)[0,1:-1]
    n = len(dy)
    pos = zeros(n)

    # position at the centre of the grid cell
    pos[0] = -0.5*dy[1]
    pos[1] = 0.5*dy[1]
    for i in range(2,n):
        pos[i] = pos[i-1] + 0.5*dy[i-1] + 0.5*dy[i]
    
    # Calculate energy transport
    flux_pos, conduction = heat_conduction(pos, Te)
    _, convect_ke = ke_convection(pos, Ne, Vi)
    _, convect_therm = thermal_convection(pos, P, Vi)

    if kappa_epar is None:
        conduction = zeros(len(flux_pos))
    
    # Return results as a dictionary
    return {"position":flux_pos,
            "conduction":conduction,
            "convection_ke":convect_ke,
            "convection_thermal":convect_therm}

def set_matplotlib_defaults():
    fontsize = 14
    plt.rc('font', size=fontsize) #controls default text size
    plt.rc('axes', titlesize=fontsize) #fontsize of the title
    plt.rc('axes', labelsize=fontsize) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=fontsize) #fontsize of the y tick labels
    plt.rc('legend', fontsize=fontsize) #fontsize of the legend
    plt.rc('lines', linewidth=3)
    plt.rc('figure', figsize=(5,5))
    plt.rc('axes', grid = True)


