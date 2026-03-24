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

    def interpolate(self, param, regions, plot_lines=False, plot_interp_debug=False):
        m = self.ds.metadata
        ds = self.ds
        solps = self.solps

        if plot_lines:
            fig, plot_line_ax = plt.subplots(figsize=(10, 20), dpi=100)
            ds["Ne"].hermesm.clean_guards().bout.polygon(
                ax=plot_line_ax,
                grid_only=True,
                linecolor="k",
                linewidth=0.3,
                antialias=True,
                separatrix=False,
            )
            plot_line_ax.set_xlim(0.1, 0.65)
            # plot_line_ax.set_ylim(-0.9, -0.5)
            plot_line_ax.set_ylim(0.5, 0.9)
            plot_line_ax.set_title("Lines=hermes, points=solps")

        params = ["R", "Z", param]
        param_values = np.zeros_like(ds[param].values)

        for region in regions:
            print(region, "\n")

            ## Get radial slice to determine sepdist for a given sepadd
            if "inner" in region:
                midplane_region = "imp"
            elif "outer" in region:
                midplane_region = "omp"
            else:
                raise ValueError(
                    f"Invalid region {region}, must contain 'inner' or 'outer'"
                )

            # Get radial slice at target to get separatrix distances
            radial = get_1d_radial_data(
                self.ds, params=["R", "Z"], guards=True, region=midplane_region
            )
            radial_sol = radial[radial["region"] == "sol"]
            num_solrings = len(radial_sol) - 2

            ## Get poloidal slices
            for sepadd in range(num_solrings):
                # for sepadd in [0]:
                radial_index = sepadd + m["ixseps1"]
                sepdist = radial_sol.loc[radial_index, "Srad"]
                print(f"{sepadd}/{sepdist:4f}, ", end="")

                # Fetch field lines once per SOL ring, shared across divertor/sol subregions
                flh = get_1d_poloidal_data(
                    ds, params=params + ["theta_idx"], region=region, sepadd=sepadd
                )
                fls = solps.get_1d_poloidal_data(
                    params=params, region=region, sepdist=sepdist
                )

                for subregion in ["divertor", "sol"]:
                    poloidal_indices, param_interp = self._get_interpolated(
                        flh,
                        fls,
                        param,
                        subregion=subregion,
                        interp_debug=plot_interp_debug,
                        plot_line_ax=plot_line_ax if plot_lines else None,
                    )

                    param_values[radial_index, poloidal_indices] = param_interp

            print("")

        ds[f"{param}_interp"] = ds[param].copy(data=param_values)

        return ds

    def _get_interpolated(
        self, flh, fls, param, subregion, interp_debug=False, plot_line_ax=None
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
            df = df.iloc[::-1]
            return df

        if subregion == "divertor":
            # Flip the field line to start at the target and
            # interpolate up to the X-point

            flh = flh[flh["region"] == "divertor"]
            fls = fls[fls["region"] == "divertor"]

            # Now target first
            flh = flip(flh)
            fls = flip(fls)

            out = scipy.interpolate.make_interp_spline(fls["Spol"], fls[param])(
                flh["Spol"]
            )

        elif subregion == "sol":
            # Do not flip, interpolate from midplane to X-point

            # Remove first point, it's at the midplane cell edge
            flh = flh.loc[1:, :]
            fls = fls.loc[1:, :]

            flh = flh[flh["region"] == "upstream"]
            fls = fls[fls["region"] == "upstream"]

            out = np.interp(flh["Spol"], fls["Spol"], fls[param])

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
