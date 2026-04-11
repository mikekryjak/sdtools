from hermes3.selectors import get_1d_poloidal_data, get_1d_radial_data
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np


class interpolateSOLPStoHermes:
    """
    WARNING: Hermes-3 must have guard cells loaded for this to work.

    """
    def __init__(self, ds, solps):
        self.ds = ds
        self.solps = solps

        self.region_settings = {
            "inner_lower_sol": dict(
                radial_start_region="imp",
                radial_subset="sol",
                segments=("divertor", "sol"),
                interpolate_midplane=True,
            ),
            "inner_upper_sol": dict(
                radial_start_region="imp",
                radial_subset="sol",
                segments=("divertor", "sol"),
                interpolate_midplane=True,
            ),
            "outer_lower_sol": dict(
                radial_start_region="omp",
                radial_subset="sol",
                segments=("divertor", "sol"),
                interpolate_midplane=True,
            ),
            "outer_upper_sol": dict(
                radial_start_region="omp",
                radial_subset="sol",
                segments=("divertor", "sol"),
                interpolate_midplane=True,
            ),
            "core": dict(
                radial_start_region="omp",
                radial_subset="core",
                segments=(None,),
                interpolate_midplane=False,
            ),
            "lower_pfr": dict(
                radial_start_region="outer_lower_target",
                radial_subset="core",
                segments=(None,),
                interpolate_midplane=False,
            ),
            "upper_pfr": dict(
                radial_start_region="outer_upper_target",
                radial_subset="core",
                segments=(None,),
                interpolate_midplane=False,
            ),
            "inner_lower_pfr": dict(
                radial_start_region="inner_lower_target",
                radial_subset="core",
                segments=(None,),
                interpolate_midplane=False,
            ),
            "outer_lower_pfr": dict(
                radial_start_region="outer_lower_target",
                radial_subset="core",
                segments=(None,),
                interpolate_midplane=False,
            ),
            "inner_upper_pfr": dict(
                radial_start_region="inner_upper_target",
                radial_subset="core",
                segments=(None,),
                interpolate_midplane=False,
            ),
            "outer_upper_pfr": dict(
                radial_start_region="outer_upper_target",
                radial_subset="core",
                segments=(None,),
                interpolate_midplane=False,
            ),
        }

        self.polygon_settings = dict(grid_only=True,
                linecolor="k",
                linewidth=0.3,
                antialias=True,
                separatrix=False,)
        
        self.polygon_xlim = (0.1, 0.65)
        self.polygon_ylim = (0.5, 0.9)

        print("\n\n CHECK GUARDS \n\n")

    def interpolate_sol(self, param, regions, plot_lines=False, plot_interp_debug=False):
        return self.interpolate(
            param,
            regions,
            plot_lines=plot_lines,
            plot_interp_debug=plot_interp_debug,
        )
    
    def interpolate_core_pfr(self, param, regions, plot_lines=False, plot_interp_debug=False):
        return self.interpolate(
            param,
            regions,
            plot_lines=plot_lines,
            plot_interp_debug=plot_interp_debug,
        )

    def interpolate(self, param, regions, plot_lines=False, plot_interp_debug=False):
        ds = self.ds

        if plot_lines:
            _, plot_line_ax = plt.subplots(figsize=(10, 20), dpi=100)
            ds["Ne"].hermesm.clean_guards().bout.polygon(
                ax=plot_line_ax,
                **self.polygon_settings,
            )
            plot_line_ax.set_xlim(*self.polygon_xlim)
            plot_line_ax.set_ylim(*self.polygon_ylim)
        else:
            plot_line_ax = None

        params = ["R", "Z", param]
        param_values = np.zeros_like(ds[param].values)

        for region in regions:
            spec = self._get_region_settings(region)
            print(region, "\n")

            for radial_index, sepadd, sepdist in self._iter_region_rings(spec):
                print(f"{radial_index}/{sepadd}/{sepdist:4f}, ", end="")

                try:
                    flh, fls = self._get_field_lines(region, spec, params, sepadd, sepdist)

                    for subregion in spec["segments"]:
                        poloidal_indices, param_interp = self._get_interpolated(
                            flh,
                            fls,
                            param,
                            subregion=subregion,
                            interp_debug=plot_interp_debug,
                            plot_line_ax=plot_line_ax,
                        )
                        param_values[radial_index, poloidal_indices] = param_interp

                except Exception as e:
                    print(
                        f"\nError interpolating {param} for region {region}, sepadd {sepadd}: {e}"
                    )

            print("")

        ds[f"{param}_interp"] = ds[param].copy(data=param_values)

        return ds

    def _get_region_settings(self, region):
        if region not in self.region_settings:
            raise ValueError(f"Invalid region {region}")

        return self.region_settings[region]

    def _iter_region_rings(self, spec):
        m = self.ds.metadata
        radial = get_1d_radial_data(
            self.ds,
            params=["R", "Z"],
            guards=True,
            region=spec["radial_start_region"],
        )
        radial_subset = radial[radial["region"] == spec["radial_subset"]]

        for radial_index in radial_subset.index.values:
            sepadd = radial_index - m["ixseps1"]
            sepdist = radial_subset.loc[radial_index, "Srad"]
            yield radial_index, sepadd, sepdist

    def _get_field_lines(self, region, spec, params, sepadd, sepdist):
        interpolate_midplane = spec["interpolate_midplane"]

        flh = get_1d_poloidal_data(
            self.ds,
            params=params + ["theta_idx"],
            region=region,
            sepadd=sepadd,
            interpolate_midplane=interpolate_midplane,
        )
        fls = self.solps.get_1d_poloidal_data(
            params=params,
            region=region,
            sepdist=sepdist,
            interpolate_midplane=interpolate_midplane,
            radial_start_region=spec["radial_start_region"],
        )

        return flh, fls

    def _interpolate_values(self, x_source, y_source, x_target):
        x_source = np.asarray(x_source)
        y_source = np.asarray(y_source)
        x_target = np.asarray(x_target)

        if len(x_source) < 2:
            raise ValueError("Need at least two points for interpolation")

        spline_order = min(3, len(x_source) - 1)
        return scipy.interpolate.make_interp_spline(x_source, y_source, k=spline_order)(
            x_target
        )

    def _get_interpolated(
        self, flh, fls, param, subregion=None, interp_debug=False, plot_line_ax=None
    ):
        """
        Interpolates a specified SOLPS parameter onto the Hermes-3 grid by performing
        1D poloidal interpolation. Supports two modes of operation: divertor and SOL
        (Scrape-Off Layer).
        Parameters
        ----------
        flh : DataFrame
            Hermes-3 field line data for the current SOL ring and region (pre-fetched).
        fls : DataFrame
            SOLPS field line data for the current SOL ring and region (pre-fetched).
        param : str
            Name of the parameter to interpolate
        subregion : str
            Interpolation subregion. Either "divertor" or "sol".
            - "divertor": Flips field lines to start at target, interpolates toward X-point.
            - "sol": Interpolates from midplane to X-point without flipping.
        interp_debug : bool, optional
            If True, generates a plot comparing Hermes, SOLPS, and interpolated data.
            Default is False.
        Returns
        -------
        tuple
            - poloidal_indices : ndarray
                Integer array of poloidal theta indices from Hermes-3 grid.
            - out : ndarray
                1D array of interpolated parameter values on Hermes-3 grid coordinates.
        Notes
        -----
        The function uses cubic spline interpolation (k=3) for divertor mode and linear
        interpolation for SOL mode. The first point is removed in SOL mode as it
        represents a midplane cell edge.
        """

        # Work on copies so the caller's DataFrames are not mutated between subregion calls
        flh = flh.copy()
        fls = fls.copy()

        def flip(df):
            df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
            df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
            df = df.iloc[::-1].reset_index(drop=True)
            return df

        if subregion == "divertor":
            # Flip the field line to start at the target and
            # interpolate up to the X-point

            flh = flh[flh["region"] == "divertor"]
            fls = fls[fls["region"] == "divertor"]

            # Now target first
            flh = flip(flh)
            fls = flip(fls)


        elif subregion == "sol":
            # Do not flip, interpolate from midplane to X-point

            # Remove first point, it's at the midplane cell edge
            flh = flh.loc[1:, :]
            fls = fls.loc[1:, :]

            flh = flh[flh["region"] == "upstream"]
            fls = fls[fls["region"] == "upstream"]

        out = self._interpolate_values(fls["Spol"], fls[param], flh["Spol"])

        if interp_debug:
            fig, ax = plt.subplots()
            ax.plot(flh["Spol"], flh[param], label="Hermes", marker="o")
            ax.plot(fls["Spol"], fls[param], label="SOLPS", marker="x")
            ax.plot(flh["Spol"], out, label="Interpolated", marker="o")
            ax.legend()

            if subregion == "sol":
                ax.set_xlabel("Poloidal distance from midplane")
            elif subregion == "divertor":
                ax.set_xlabel("Poloidal distance from target")
            # ax.set_xlim(0.8, None)

        if plot_line_ax:
            plot_line_ax.plot(flh["R"], flh["Z"], lw=1)
            plot_line_ax.plot(fls["R"], fls["Z"], lw=0, marker="o", c="deeppink", ms=1)

        poloidal_indices = flh["theta_idx"].values.astype(int)

        return poloidal_indices, out
    

    
    def check_1d(self, ds_interp, param):

        lw = 1
        style_SOLPS = dict(c="k", marker="o", linewidth = lw, label="SOLPS")
        style_Hermes = dict(c="darkorange", marker="+", linewidth = lw, label="Hermes")
        style_interp = dict(c="deeppink", marker="X", linewidth = lw, label="Hermes interpolated")

        def plot_radial_result(ax, param, solps_region, hermes_region):

            hermes = get_1d_radial_data(ds_interp, params = [param], region = hermes_region)
            hermes_interp = get_1d_radial_data(ds_interp, params = [f"{param}_interp"], region = hermes_region)
            solps = self.solps.get_1d_radial_data(params = [param], region = solps_region)

            ax.plot(solps["dist"], solps[param], **style_SOLPS)
            ax.plot(hermes["Srad"], hermes[param], **style_Hermes)
            ax.plot(hermes_interp["Srad"], hermes_interp[f"{param}_interp"], **style_interp)
            
            ax.set_title(hermes_region)
            ax.legend(fontsize = "x-small")

        def plot_parallel_result(ax, param, sepdist, solps_region, hermes_region):
            
            hermes = get_1d_poloidal_data(ds_interp, params = [param], sepdist = sepdist, region = hermes_region)
            hermes_interp = get_1d_poloidal_data(ds_interp, params = [f"{param}_interp"], sepdist = sepdist, region = hermes_region)
            solps = self.solps.get_1d_poloidal_data(params = [param], sepdist = sepdist, region = solps_region)

            ax.plot(solps["Spar"], solps[param], **style_SOLPS)
            ax.plot(hermes["Spar"], hermes[param], **style_Hermes)
            ax.plot(hermes_interp["Spar"], hermes_interp[f"{param}_interp"], **style_interp)
            
            ax.set_title(f"{hermes_region}, sepdist={sepdist}")
            ax.legend(fontsize = "x-small")

        num_plots = 5
        fig, axes = plt.subplots(2,num_plots, figsize=(num_plots*5,10))

        plot_radial_result(axes[0,0], "Ne", "omp", "omp")
        plot_radial_result(axes[0,1], "Ne", "imp", "imp")
        plot_radial_result(axes[0,2], "Ne", "inner_lower_target", "inner_lower_target")
        plot_radial_result(axes[0,3], "Ne", "outer_lower_target", "outer_lower_target")

        # These sepdist values are at Hermes-3 cell centres
        plot_parallel_result(axes[1,0], "Ne", 0.000202, "outer_lower_sol", "outer_lower_sol")
        plot_parallel_result(axes[1,1], "Ne", 0.014630, "outer_lower_sol", "outer_lower_sol")
        plot_parallel_result(axes[1,2], "Ne", 0.000202, "inner_lower_sol", "inner_lower_sol")
        plot_parallel_result(axes[1,3], "Ne", 0.014630, "inner_lower_sol", "inner_lower_sol")
