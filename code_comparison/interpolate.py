from hermes3.selectors import get_1d_poloidal_data, get_1d_radial_data
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np


def read_geqdsk(content):
    """
    Minimal G-EQDSK reader returning the poloidal flux map and magnetic axis.

    Only the fields needed to define flux-surface geometry are extracted. The
    float block is read with fixed 16-character columns because adjacent negative
    values in G-EQDSK are written with no separating space (so str.split() fails).

    Parameters
    ----------
    content : str
        Full text of a G-EQDSK file.

    Returns
    -------
    dict with:
        R1d, Z1d : 1D grids of the flux map (m)
        psi      : 2D poloidal flux on (Z, R), shape (nh, nw)
        rmaxis, zmaxis : magnetic axis (m)
        simag, sibry   : psi at axis and at the plasma boundary (separatrix)
    """
    lines = content.splitlines()
    nw, nh = int(lines[0].split()[-2]), int(lines[0].split()[-1])

    # Number of floats before the boundary section: 4 header lines (20 values) +
    # fpol, pres, ffprim, pprime (nw each) + psi(nw*nh) + qpsi(nw).
    need = 20 + 4 * nw + nw * nh + nw
    vals = []
    for line in lines[1:]:
        for k in range(0, len(line), 16):
            chunk = line[k : k + 16]
            if chunk.strip():
                vals.append(float(chunk))
        if len(vals) >= need:
            break
    v = np.array(vals[:need])

    rdim, zdim, rcentr, rleft, zmid = v[0:5]
    rmaxis, zmaxis, simag, sibry, bcentr = v[5:10]
    psi = v[20 + 4 * nw : 20 + 4 * nw + nw * nh].reshape(nh, nw)  # (Z, R)

    return dict(
        R1d=rleft + np.linspace(0, rdim, nw),
        Z1d=zmid + np.linspace(-zdim / 2, zdim / 2, nh),
        psi=psi,
        rmaxis=rmaxis,
        zmaxis=zmaxis,
        simag=simag,
        sibry=sibry,
    )


def project_to_polyline(points, poly):
    """
    Project points onto an ordered polyline.

    For each query point return the arc-length position of its nearest foot-point
    along `poly`, and the perpendicular distance to `poly`. This is how both the
    Hermes-3 and SOLPS field-line points are assigned a common poloidal coordinate
    along a shared equilibrium flux-surface contour.

    Parameters
    ----------
    points : (N, 2) array of (R, Z)
    poly   : (M, 2) ordered array of (R, Z) defining the reference curve

    Returns
    -------
    s    : (N,) arc-length coordinate along `poly` of each point's foot-point
    dist : (N,) perpendicular distance from each point to `poly`
    """
    points = np.asarray(points)
    a = poly[:-1]  # segment start points
    ab = poly[1:] - a  # segment vectors
    ab2 = np.einsum("ij,ij->i", ab, ab)
    ab2 = np.where(ab2 == 0, 1e-30, ab2)
    seglen = np.sqrt(np.einsum("ij,ij->i", ab, ab))
    cumlen = np.concatenate([[0.0], np.cumsum(seglen)])

    s = np.empty(len(points))
    dist = np.empty(len(points))
    for i, p in enumerate(points):
        t = np.clip(((p - a) * ab).sum(1) / ab2, 0.0, 1.0)  # foot param per segment
        foot = a + t[:, None] * ab
        d = np.hypot(*(p - foot).T)
        j = int(np.argmin(d))
        s[i] = cumlen[j] + t[j] * seglen[j]
        dist[i] = d[j]
    return s, dist


class interpolateSOLPStoHermes:
    """
    Interpolate a SOLPS solution onto a Hermes-3 grid, field line by field line.

    Every region (SOL legs, core, PFR) is interpolated on a coordinate built from
    the shared equilibrium flux surfaces (see ``_equilibrium_coords``): each point
    is projected onto a high-resolution contour of the true psi, so the X-point and
    connection lengths line up between the two codes and no X-point split is needed.
    This requires ``equilibrium=...``; the legacy raw-poloidal-distance ("poloidal")
    coordinate remains available per-region as a fallback.

    WARNING: the Hermes-3 dataset must be loaded WITH guard cells.
    """

    def __init__(self, ds, solps, equilibrium=None):
        """
        Parameters
        ----------
        ds : Hermes-3 dataset (must be loaded WITH guard cells).
        solps : SOLPScase to interpolate from.
        equilibrium : str, optional
            Path to the shared equilibrium, either a G-EQDSK file or a BOUT++ grid
            ``.nc`` that embeds ``hypnotoad_input_geqdsk_file_contents``. When given,
            SOL legs are interpolated against the true flux-surface geometry (see
            ``_equilibrium_coords``) instead of raw poloidal distance. Both codes
            share this equilibrium, so one map serves both.
        """
        self.ds = ds
        self.solps = solps

        # Load the equilibrium flux map used to build the poloidal coordinate.
        self.eq = None
        if equilibrium is not None:
            self._load_equilibrium(equilibrium)

        # Each entry describes how one region's field lines are built and on which
        # poloidal coordinate the SOLPS->Hermes interpolation is performed:
        #   poloidal_region     - region passed to get_1d_poloidal_data
        #   radial_start_region - radial slice defining the separatrix distance
        #   radial_subset       - "sol" (positive sepdist) or "core" (negative)
        #   coordinate          - "equilibrium" (project onto flux-surface contour)
        #                         or "poloidal"  (legacy: raw poloidal distance Spol)
        #   interpolate_midplane- anchor the field line exactly at the midplane (Z=0)
        #   flip                - reverse so the target is first (poloidal coord only)
        #
        # SOL legs run the FULL leg (midplane -> target) as a single piece. The
        # X-point is no longer a breakpoint: it falls at a consistent location in
        # both codes via the equilibrium coordinate, so no split is needed.
        def sol_leg(start):
            return dict(
                radial_start_region=start,
                radial_subset="sol",
                coordinate="equilibrium",
                interpolate_midplane=True,
                flip=False,
            )

        def pfr_leg(start):
            return dict(
                radial_start_region=start,
                radial_subset="core",
                coordinate="equilibrium",
                interpolate_midplane=False,
                flip=False,
            )

        self.region_settings = {
            "outer_lower": sol_leg("omp"),
            "outer_upper": sol_leg("omp"),
            "inner_lower": sol_leg("imp"),
            "inner_upper": sol_leg("imp"),
            "core": dict(
                radial_start_region="omp",
                radial_subset="core",
                coordinate="equilibrium",
                interpolate_midplane=False,
                flip=False,
            ),
            "inner_lower_pfr": pfr_leg("inner_lower_target"),
            "outer_lower_pfr": pfr_leg("outer_lower_target"),
            "inner_upper_pfr": pfr_leg("inner_upper_target"),
            "outer_upper_pfr": pfr_leg("outer_upper_target"),
        }
        # The region passed to get_1d_poloidal_data: SOL legs need the "_sol" suffix.
        for name, spec in self.region_settings.items():
            spec["poloidal_region"] = (
                f"{name}_sol" if spec["radial_subset"] == "sol" else name
            )

        self.polygon_settings = dict(
            grid_only=True,
            linecolor="k",
            linewidth=0.3,
            antialias=True,
            separatrix=False,
        )

        # self.polygon_xlim = (0.1, 0.65)
        # self.polygon_ylim = (0.5, 0.9)

        self.polygon_xlim = (None, None)
        self.polygon_ylim = (None, None)

    def _load_equilibrium(self, path):
        """
        Load the equilibrium psi map and build the tools used to make the poloidal
        coordinate: a smooth psi(R,Z) spline and a pre-evaluated fine (R,Z) grid for
        contouring flux surfaces.

        `path` is either a G-EQDSK file or a BOUT++ grid ``.nc`` embedding
        ``hypnotoad_input_geqdsk_file_contents``.
        """
        if path.endswith(".nc"):
            from boututils.datafile import DataFile

            with DataFile(path) as gridfile:
                raw = np.asarray(
                    gridfile["hypnotoad_input_geqdsk_file_contents"]
                ).item()
            content = raw if isinstance(raw, str) else raw.decode()
        else:
            with open(path) as f:
                content = f.read()

        self.eq = read_geqdsk(content)
        self.psi_spline = scipy.interpolate.RectBivariateSpline(
            self.eq["R1d"], self.eq["Z1d"], self.eq["psi"].T
        )
        # Fine (R,Z) evaluation of psi for high-resolution flux-surface contours.
        self._eq_Rf = np.linspace(self.eq["R1d"][0], self.eq["R1d"][-1], 600)
        self._eq_Zf = np.linspace(self.eq["Z1d"][0], self.eq["Z1d"][-1], 900)
        self._eq_PSI = self.psi_spline(self._eq_Rf, self._eq_Zf).T  # (Z, R)

    def interpolate(self, param, regions, plot_lines=False, plot_interp_debug=False):
        ds = self.ds
        self.plot_lines = plot_lines

        if plot_lines:
            _, plot_line_ax = plt.subplots(figsize=(10, 20), dpi=100)
            ds["Ne"].hermes.clear_guards().bout.polygon(
                ax=plot_line_ax,
                **self.polygon_settings,
            )

            self.plot_line_ax = plot_line_ax

            self.solps.plot_2d("Ne", ax=plot_line_ax, grid_only=True, linecolor="red")
            plot_line_ax.set_xlim(*self.polygon_xlim)
            plot_line_ax.set_ylim(*self.polygon_ylim)
        else:
            plot_line_ax = None

        params = ["R", "Z", param]
        param_values = np.zeros_like(ds[param].values)

        for region in self._expand_regions(regions):
            spec = self._get_region_settings(region)
            print(region, "\n")

            if plot_line_ax is not None:
                self._plot_invalid_rings(spec, plot_line_ax)

            for radial_index, sepadd, sepdist in self._iter_region_rings(spec):
                print(f"{radial_index}/{sepadd}/{sepdist:4f}, ", end="")

                flh, fls = self._get_field_lines(spec, params, sepadd, sepdist)

                poloidal_indices, param_interp = self._get_interpolated(
                    flh,
                    fls,
                    param,
                    region=region,
                    spec=spec,
                    interp_debug=plot_interp_debug,
                    plot_line_ax=plot_line_ax,
                )
                param_values[radial_index, poloidal_indices] = param_interp

            print("")

        ds[f"{param}_interp"] = ds[param].copy(data=param_values)

        return ds

    def _get_region_settings(self, region):
        if region not in self.region_settings:
            raise ValueError(f"Invalid region {region}")

        return self.region_settings[region]

    def _expand_regions(self, regions):
        """Accept either a bare leg name ('outer_lower') or its '*_sol' form."""
        return [r[:-4] if r.endswith("_sol") else r for r in regions]

    def _get_region_ring_sets(self, spec):
        hermes_radial_slice = get_1d_radial_data(
            self.ds,
            params=["R", "Z"],
            guards=False,
            region=spec["radial_start_region"],
            debug=False,
        )

        solps_radial_slice = self.solps.get_1d_radial_data(
            ["R", "Z"],
            guards=False,
            region=spec["radial_start_region"],
        )

        if getattr(self, "plot_lines", False):
            self.plot_line_ax.plot(
                hermes_radial_slice["R"],
                hermes_radial_slice["Z"],
                lw=2,
                marker="o",
                c="cyan",
                ms=3,
            )

        if spec["radial_subset"] == "sol":
            hermes_candidate_rings = hermes_radial_slice[
                hermes_radial_slice["Srad"] > 0
            ]
            solps_available_rings = solps_radial_slice[solps_radial_slice["dist"] > 0]
        elif spec["radial_subset"] == "core":
            hermes_candidate_rings = hermes_radial_slice[
                hermes_radial_slice["Srad"] < 0
            ]
            solps_available_rings = solps_radial_slice[solps_radial_slice["dist"] < 0]
        else:
            raise ValueError(f"Unsupported radial subset {spec['radial_subset']}")

        if len(solps_available_rings) == 0:
            valid_hermes_rings = hermes_candidate_rings.iloc[0:0]
            invalid_hermes_rings = hermes_candidate_rings
        else:
            valid_hermes_rings = hermes_candidate_rings
            invalid_hermes_rings = hermes_candidate_rings.iloc[0:0]

        return valid_hermes_rings, invalid_hermes_rings

    def _plot_invalid_rings(self, spec, plot_line_ax):
        m = self.ds.metadata
        _, invalid_hermes_rings = self._get_region_ring_sets(spec)

        for radial_index in invalid_hermes_rings.index.values:
            sepadd = radial_index - m["ixseps1"]
            invalid_field_line = get_1d_poloidal_data(
                self.ds,
                params=["R", "Z", "theta_idx"],
                region=spec["poloidal_region"],
                sepadd=sepadd,
                radial_start_region=spec["radial_start_region"],
                interpolate_midplane=spec["interpolate_midplane"],
                debug=False,
            )
            plot_line_ax.plot(
                invalid_field_line["R"],
                invalid_field_line["Z"],
                lw=2,
                marker="o",
                ms=6,
                c="red",
                markerfacecolor="black",
            )

    def _iter_region_rings(self, spec):
        """
        Iterable to return valid Hermes-3 poloidal rings and the corresponding
        sepadd and sepdist. Valid means there is SOLPS data available at the desired
        Hermes-3 sepdist.
        """
        m = self.ds.metadata
        valid_hermes_rings, _ = self._get_region_ring_sets(spec)

        for radial_index in valid_hermes_rings.index.values:
            # for radial_index in [20]:
            sepadd = radial_index - m["ixseps1"]
            sepdist = valid_hermes_rings.loc[radial_index, "Srad"]
            yield radial_index, sepadd, sepdist

    def _get_field_lines(self, spec, params, sepadd, sepdist):
        """
        Return the matched Hermes-3 and SOLPS field-line DataFrames for one ring.

        Both are taken on the same flux surface: Hermes-3 by ring index (sepadd),
        SOLPS by radially interpolating to the same separatrix distance (sepdist).
        """
        interpolate_midplane = spec["interpolate_midplane"]
        hermes_region = spec["poloidal_region"]

        flh = get_1d_poloidal_data(
            self.ds,
            params=params + ["theta_idx"],
            region=hermes_region,
            sepadd=sepadd,
            radial_start_region=spec["radial_start_region"],
            interpolate_midplane=interpolate_midplane,
            debug=False,
        )

        # SOLPS needs the "_extra" region (one cell past the midplane) to anchor
        # exactly at Z=0 when interpolate_midplane is on.
        solps_region = (
            f"{hermes_region}_extra" if interpolate_midplane else hermes_region
        )
        fls = self.solps.get_1d_poloidal_data(
            params=params,
            region=solps_region,
            sepdist=sepdist,
            interpolate_midplane=interpolate_midplane,
            radial_start_region=spec["radial_start_region"],
            extrapolate_radial=True,
            debug=False,
            guards=False,
        )

        return flh, fls

    def _interpolate_values(self, x_source, y_source, x_target):
        x_source = np.asarray(x_source)
        y_source = np.asarray(y_source)
        x_target = np.asarray(x_target)

        if len(x_source) < 2:
            raise ValueError("Need at least two points for interpolation")

        x_min = x_source.min()
        x_max = x_source.max()
        below_source = x_target < x_min
        above_source = x_target > x_max
        num_extrapolated = np.count_nonzero(below_source | above_source)

        if num_extrapolated > 0:
            print(
                "Extrapolating poloidal interpolation for "
                f"{num_extrapolated} target points outside the source range "
                f"[{x_min}, {x_max}]"
            )

        return scipy.interpolate.interp1d(
            x_source,
            y_source,
            kind="linear",
            bounds_error=False,
            # fill_value="extrapolate",
            fill_value=(y_source[0], y_source[-1]),
        )(x_target)

    @staticmethod
    def _flip_field_line(df):
        """Reverse a field line so the target is first (poloidal-coordinate path)."""
        df = df.copy()
        df["Spol"] = df["Spol"].iloc[-1] - df["Spol"]
        df["Spar"] = df["Spar"].iloc[-1] - df["Spar"]
        return df.iloc[::-1].reset_index(drop=True)

    def _get_interpolated(
        self, flh, fls, param, region, spec, interp_debug=False, plot_line_ax=None
    ):
        """
        Interpolate one SOLPS parameter onto the matched Hermes-3 field line.

        The poloidal coordinate used for the 1D interpolation depends on the region:

        - ``coordinate="equilibrium"`` (SOL legs): both field lines are projected onto
          a high-resolution contour of the true equilibrium flux surface, so the
          X-point lands at a consistent location in both codes (see
          ``_equilibrium_coords``). No X-point split, no constant-value extrapolation
          except over the small genuine target-geometry overhang.
        - ``coordinate="poloidal"`` (core / PFR): the legacy raw poloidal distance
          ``Spol``, optionally flipped so the target is first.

        Returns
        -------
        poloidal_indices : ndarray of int
            Hermes-3 theta indices to write to (the midplane-anchored point, which
            has no cell, is excluded).
        out : ndarray
            Interpolated values at those indices.
        """
        flh = flh.copy()
        fls = fls.copy()

        if len(flh) < 2 or len(fls) < 2:
            raise ValueError(f"Not enough points to interpolate {region}")

        if spec["coordinate"] == "equilibrium":
            if self.eq is None:
                raise ValueError(
                    "coordinate='equilibrium' requires an equilibrium; pass "
                    "`equilibrium=...` to interpolateSOLPStoHermes"
                )
            x_h, x_s = self._equilibrium_coords(flh, fls)
            xlabel = "Equilibrium arc length [m]"
        else:
            if spec["flip"]:
                flh = self._flip_field_line(flh)
                fls = self._flip_field_line(fls)
            x_h, x_s = flh["Spol"].values, fls["Spol"].values
            xlabel = "Poloidal distance [m]"

        # interp1d needs a monotonic source; sort the SOLPS points by the coordinate.
        order = np.argsort(x_s)
        out = self._interpolate_values(
            np.asarray(x_s)[order], fls[param].values[order], x_h
        )

        if interp_debug:
            _, ax = plt.subplots()
            ax.plot(x_h, flh[param], label="Hermes", marker="o")
            ax.plot(x_s, fls[param], label="SOLPS", marker="x", c="k")
            ax.plot(x_h, out, label="Interpolated", c="darkorange", marker="o")
            ax.set_xlabel(xlabel)
            ax.set_title(region)
            ax.legend()

        if plot_line_ax:
            plot_line_ax.plot(fls["R"], fls["Z"], marker="o", lw=2, ms=3, c="deeppink")

        # Write back only real cells: the midplane-anchored point has theta_idx=NaN.
        theta = flh["theta_idx"].values
        valid = ~np.isnan(theta)
        return theta[valid].astype(int), out[valid]

    def _equilibrium_coords(self, flh, fls):
        """
        Common poloidal coordinate for a full SOL leg, from the shared equilibrium.

        After radial matching, the Hermes-3 (``flh``) and SOLPS (``fls``) field lines
        lie on the same flux surface. Both are projected onto ONE high-resolution
        contour of the equilibrium psi at this leg's psi level, so the resulting
        arc-length coordinate is grid-independent and the X-point coincides in both
        codes. Returns ``(s_flh, s_fls)`` arc-length coordinates.
        """
        import contourpy

        flhRZ = flh[["R", "Z"]].values
        flsRZ = fls[["R", "Z"]].values

        # psi level of this leg, evaluated from the Hermes points (geqdsk sign
        # convention - never compared against Hermes psi, so sign-agnostic).
        psi_level = float(np.median(self.psi_spline.ev(flhRZ[:, 0], flhRZ[:, 1])))

        # contourpy returns each connected flux-surface contour as an ORDERED line.
        cg = contourpy.contour_generator(self._eq_Rf, self._eq_Zf, self._eq_PSI)
        lines = cg.lines(psi_level)

        # Pick the single contour carrying this leg (smallest median distance to the
        # Hermes points). Using one ordered line avoids pulling in neighbouring
        # branches that pass close by near the X-point.
        median_dist = [np.median(project_to_polyline(flhRZ, L)[1]) for L in lines]
        contour = lines[int(np.argmin(median_dist))]

        # Trim the contour to the arc spanned by the Hermes leg (+ small margin), so
        # far parts of the same flux surface can't capture points near the X-point.
        s_on_contour, _ = project_to_polyline(flhRZ, contour)
        seglen = np.hypot(*np.diff(contour, axis=0).T)
        s_contour = np.concatenate([[0.0], np.cumsum(seglen)])
        lo, hi = s_on_contour.min(), s_on_contour.max()
        margin = 0.05 * (hi - lo)
        reference = contour[(s_contour >= lo - margin) & (s_contour <= hi + margin)]

        s_flh, _ = project_to_polyline(flhRZ, reference)
        s_fls, _ = project_to_polyline(flsRZ, reference)

        # The equilibrium arc length already aligns the X-point between the two
        # codes, but the two grids' leg ENDS (target, midplane) sit at slightly
        # different physical positions (different wall/cell layout, ~mm at the
        # target). Affine-normalise each leg's coordinate to [0, 1] so both
        # endpoints coincide; the rescale is tiny (the leg-length difference, well
        # under a percent) and linear, so it does not distort the profile or move
        # the X-point, but it removes the systematic target offset.
        def normalise(s):
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo) if hi > lo else s

        return normalise(s_flh), normalise(s_fls)

    def plot_1d_check(self, ds_interp, param):
        """
        Series of 1D plots to check interpolation quality for a single parameter
        """

        lw = 1
        style_SOLPS = dict(c="k", marker="o", linewidth=lw, label="SOLPS")
        style_Hermes = dict(c="darkorange", marker="+", linewidth=lw, label="Hermes")
        style_interp = dict(
            c="deeppink", marker="X", linewidth=lw, label="Hermes interpolated"
        )

        def plot_radial_result(ax, param, solps_region, hermes_region):

            hermes = get_1d_radial_data(ds_interp, params=[param], region=hermes_region)
            hermes_interp = get_1d_radial_data(
                ds_interp, params=[f"{param}_interp"], region=hermes_region
            )
            solps = self.solps.get_1d_radial_data(params=[param], region=solps_region)

            ax.plot(solps["dist"], solps[param], **style_SOLPS)
            ax.plot(hermes["Srad"], hermes[param], **style_Hermes)
            ax.plot(
                hermes_interp["Srad"], hermes_interp[f"{param}_interp"], **style_interp
            )

            ax.set_title(hermes_region)
            ax.legend(fontsize="x-small")

        def plot_parallel_result(ax, param, sepadd, solps_region, hermes_region):

            radial_region = "omp" if "outer" in hermes_region else "imp"
            radial_slice = get_1d_radial_data(
                ds_interp, params=["R", "Z"], region=radial_region, guards=False
            )
            radial_sol = radial_slice[radial_slice["region"] == "sol"]
            sepdist = radial_sol.iloc[sepadd]["Srad"]

            hermes = get_1d_poloidal_data(
                ds_interp,
                params=[param],
                sepdist=sepdist,
                region=hermes_region,
                guards=False,
            )
            hermes_interp = get_1d_poloidal_data(
                ds_interp,
                params=[f"{param}_interp"],
                sepdist=sepdist,
                region=hermes_region,
                guards=False,
            )
            solps = self.solps.get_1d_poloidal_data(
                params=[param],
                sepdist=sepdist,
                region=solps_region,
                guards=False,
                debug=False,
            )

            ax.plot(solps["Spol"], solps[param], **style_SOLPS)
            ax.plot(hermes["Spol"], hermes[param], **style_Hermes)
            ax.plot(
                hermes_interp["Spol"], hermes_interp[f"{param}_interp"], **style_interp
            )

            ax.set_title(f"{hermes_region}, hermes sepadd={sepadd}")
            ax.legend(fontsize="x-small")

        num_cols = 6
        fig, axes = plt.subplots(3, num_cols, figsize=(num_cols * 5, 15))

        plot_radial_result(axes[0, 0], param, "omp", "omp")
        plot_radial_result(axes[0, 1], param, "imp", "imp")
        plot_radial_result(
            axes[0, 2], param, "inner_lower_target", "inner_lower_target"
        )
        plot_radial_result(
            axes[0, 3], param, "inner_upper_target", "inner_upper_target"
        )
        plot_radial_result(
            axes[0, 4], param, "outer_lower_target", "outer_lower_target"
        )
        plot_radial_result(
            axes[0, 5], param, "outer_upper_target", "outer_upper_target"
        )

        # These sepdist values are at Hermes-3 cell centres
        plot_parallel_result(
            axes[1, 2], param, 0, "inner_lower_sol_extra", "inner_lower_sol"
        )
        plot_parallel_result(
            axes[1, 3], param, 0, "inner_upper_sol_extra", "inner_upper_sol"
        )
        plot_parallel_result(
            axes[1, 4], param, 0, "outer_lower_sol_extra", "outer_lower_sol"
        )
        plot_parallel_result(
            axes[1, 5], param, 0, "outer_upper_sol_extra", "outer_upper_sol"
        )

        plot_parallel_result(
            axes[2, 2], param, 4, "inner_lower_sol_extra", "inner_lower_sol"
        )
        plot_parallel_result(
            axes[2, 3], param, 4, "inner_upper_sol_extra", "inner_upper_sol"
        )
        plot_parallel_result(
            axes[2, 4], param, 4, "outer_lower_sol_extra", "outer_lower_sol"
        )
        plot_parallel_result(
            axes[2, 5], param, 4, "outer_upper_sol_extra", "outer_upper_sol"
        )
