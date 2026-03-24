from hermes3.selectors import get_1d_poloidal_data, get_1d_radial_data
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np


class interpolateSOLPStoHermes:
    def __init__(self, ds, solps):
        self.ds = ds
        self.solps = solps

    def interpolate(self, param, regions, plot_lines=False):
        m = self.ds.metadata
        ds = self.ds
        solps = self.solps

        # Get radial slice at target to get separatrix distances
        radial = get_1d_radial_data(
            self.ds, params=["R", "Z"], guards=True, region="omp"
        )
        radial_sol = radial[radial["Srad"] > 0]
        param_values = np.zeros_like(ds[param].values)
        num_solrings = m["nx"] - m["ixseps1"] - 2

        print(num_solrings)

        if plot_lines:
            fig, plot_line_ax = plt.subplots(dpi=200)
            ds["Ne"].hermesm.clean_guards().bout.polygon(
                ax=plot_line_ax,
                grid_only=True,
                linecolor="k",
                linewidth=0.3,
                antialias=True,
                separatrix=False,
            )

        for region in regions:
            for subregion in ["divertor", "sol"]:
                print(region, subregion, "\n")

                for sepadd in range(num_solrings):
                    radial_index = sepadd + m["ixseps1"]
                    print(radial_index, ", ", end="")
                    sepdist = radial_sol["Srad"][radial_index]

                    poloidal_indices, param_interp = self._get_interpolated(
                        sepadd,
                        sepdist,
                        param,
                        region=region,
                        subregion=subregion,
                        debug=False,
                        plot_line_ax=plot_line_ax if plot_lines else None,
                    )

                    for i in range(len(poloidal_indices)):
                        param_values[radial_index, poloidal_indices[i]] = param_interp[
                            i
                        ]

                print("")

        ds[f"{param}_interp"] = ds[param].copy(data=param_values)

        # return ds

    def _get_interpolated(
        self, sepadd, sepdist, param, region, subregion, debug=False, plot_line_ax=None
    ):
        """
        Interpolates a specified SOLPS parameter onto the Hermes-3 grid by performing
        1D poloidal interpolation. Supports two modes of operation: divertor and SOL
        (Scrape-Off Layer).
        Parameters
        ----------
        sepadd : float
            SOL ring selection (number of SOL rings from separatrix in Hermes-3 grid)
        sepdist : float
            Distance of selected SOL ring from separatrix (to get equivalent from SOLPS)
        param : str
            Name of the parameter to interpolate
        region : str
            outer_lower, outer_upper, inner_lower, inner_upper
        subregion : str
            Interpolation subregion. Either "divertor" or "sol".
            - "divertor": Flips field lines to start at target, interpolates toward X-point.
            - "sol": Interpolates from midplane to X-point without flipping.
        debug : bool, optional
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
        interpolation (k=1) for SOL mode. The first point is removed in SOL mode as it
        represents a midplane cell edge.
        """

        ds = self.ds
        solps = self.solps
        m = ds.metadata

        params = ["R", "Z", param]
        flh = get_1d_poloidal_data(
            ds, params=params + ["theta_idx"], region=region, sepadd=sepadd
        )
        fls = solps.get_1d_poloidal_data(params=params, region=region, sepdist=sepdist)

        # if debug:
        #     fig, ax = plt.subplots()
        #     ax.plot(flh["Spol"], flh[param], label = "Hermes", marker = "o")
        #     ax.plot(fls["Spol"], fls[param], label = "SOLPS", marker = "x")

        #     ax.legend()
        #     ax.set_title(f"Full: {sepadd}")

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

            out = scipy.interpolate.make_interp_spline(fls["Spol"], fls[param], k=1)(
                flh["Spol"]
            )

            if debug:
                fig, ax = plt.subplots()
                ax.plot(flh["Spol"], flh[param], label="Hermes", marker="o")
                ax.plot(fls["Spol"], fls[param], label="SOLPS", marker="x")
                ax.plot(flh["Spol"], out, label="Interpolated", marker="o")
                ax.legend()
                ax.set_title(sepadd)
                # ax.set_xlim(0.8, None)

        if plot_line_ax:
            plot_line_ax.plot(flh["R"], flh["Z"], lw = 1)

        poloidal_indices = flh["theta_idx"].values.astype(int)
        return poloidal_indices, out
