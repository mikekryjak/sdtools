import os
import numpy as np
import pandas as pd
from hermes3.load import *
from hermes3.utils import *

class CaseDeck1D:
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
            self.cases[case] = Load.case_1D(self.casepaths[case])
            
            suffix = case.split("-")[-1]
            self.suffix[suffix] = self.cases[case]
            
            # self.cases[case].get_htransfer()


        # self.get_stats()
        
        print("...Done")

    def get_stats(self):
        self.stats = pd.DataFrame()
        self.stats.index.rename("case", inplace = True)

        for casename in self.casenames:
            case = self.cases[casename]
            self.stats.loc[casename, "target_flux"] = case.ds["NVd+"].isel(t=-1).values[-2]
            self.stats.loc[casename, "target_temp"] = case.ds["Te"].isel(t=-1).values[-2]


            self.stats.loc[casename, "initial_dens"] = case.ds.options["Nd+"]["function"] * case.ds.metadata["Nnorm"]
            self.stats.loc[casename, "line_dens"] = case.ds.options["Nd+"]["function"] * case.ds.metadata["Nnorm"]


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

    def plot(self, vars = [["Te", "Ne", "Nd"], 
                           ["Sd+_iz", "Rd+_ex", "Fdd+_cx"], 
                        #    ["NVi", "M", "F"]
                           ],
             trim = True, scale = "auto", xlims = (0,0)):
        # lib = library()


        colors = mike_cmap()
        lw = 2

        for list_params in vars:

            fig, axes = plt.subplots(1,3, figsize = (18,5))
            fig.subplots_adjust(wspace=0.4)

            for i, ax in enumerate(axes):
                param = list_params[i]
                for i, case in enumerate(self.casenames):
                    ds = self.cases[case].ds.isel(t=-1, pos=slice(2,-1))   # Final timestep, skip outer guards
                    
                    if param in ds.data_vars:
                        ax.plot(ds["pos"], ds[param], color = colors[i], linewidth = lw, label = case)


                ax.set_xlabel("Position (m)")
                ax.set_ylabel("{} ({})".format(ds[param].name, ds[param].units), fontsize = 11)
                
                ax.set_yscale("symlog")
                # if scale == "auto":
                    # ax.set_yscale(lib[param]["scale"])
                if scale == "log":
                    ax.set_yscale("log")
                elif scale == "symlog":
                    ax.set_yscale("symlog")
                    
                ax.set_title(param)
                ax.legend(fontsize = 10)
                ax.grid(which="major", alpha = 0.3)
                zoomlims = (max(ds["pos"].values)*0.91, max(ds["pos"].values)*1.005)
                if trim and param in ["NVi", "P", "M", "Ne", "Nn",  "Fcx", 
                "Frec", "E", "Fdd+_cx", "R", "Rex", "Rrec", "Riz", 
                "Siz", "S", "Eiz", "Vi", "Pn", "NVn"]:
                    ax.set_xlim(zoomlims)

                if xlims != (0,0):
                    ax.set_xlim(xlims)

                ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.1e}"))