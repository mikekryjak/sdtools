def loglog_cooling_curve_fit(name, T, Lz, data_limits = (2,2000), fit_limits = (1,2000), plot_output = True, print_output = True):
    """ 
    Tool to produce fixed fraction impurity curves.
    They are fit with a 10 coefficient polynomial in log-log space (like AMJUEL does).
    The data used for the fit is trimmed to within lo_limit and hi_limit.
    This allows the fit to extrapolate past the data and gently take the curve to near-zero at the extremes.
    This will artificially reduce radiation at very high temperatures which makes it a bit more realistic since the nTau choice and even the fluid approximation break down in the core.

    Parameters
    ----------
        Te[eV] and Lz[W/m3] from cooling curve data (e.g. ADAS).
        data_limits: a tuple of the low and high temperature limit beyond which to trim the data
        fit_limits: a tuple of the low and high temperature limit beyond which the function will keep the same value as at the limit

    Returns
    -------
        Print of coefficients in a way that can be copy pasted into Hermes-3. The coefficients are for a polynomial that takes in log(T) and returns log(Lz)
        Plots showing fits
    """

    df = pd.DataFrame()
    df["T"] = T
    df["Lz"] = Lz
    df = df[df["T"] < 3000]  # Not interested in values above 3000

    T_hires = np.linspace(0, 3000, 100000)

    df["logT"] = np.log(df["T"])
    logT_hires = np.log(T_hires)
    df["logLz"] = np.log(df["Lz"])

    dftrim = df[(df["T"] > data_limits[0]) & (df["T"] < data_limits[1])]

    coeffs = np.polyfit(dftrim["logT"], dftrim["logLz"], deg = 10)[::-1]    # polyfit gives coeffs in opposite order to what polynomial needs 
    fit_func = np.polynomial.polynomial.Polynomial(coeffs)
    logLz_fit = [fit_func(x) for x in logT_hires]
    Lz_fit = np.exp(logLz_fit)

    fit_lo_limit_index = np.argmin(np.abs(T_hires - fit_limits[0]))
    fit_hi_limit_index = np.argmin(np.abs(T_hires - fit_limits[1]))
    Lz_lo = Lz_fit[fit_lo_limit_index]
    Lz_hi = Lz_fit[fit_hi_limit_index]

    Lz_fit_trim = np.where(T_hires > fit_limits[0], Lz_fit, Lz_lo)
    Lz_fit_trim = np.where(T_hires < fit_limits[1], Lz_fit_trim, Lz_hi)

    if plot_output is True:
        fig, axes = plt.subplots(3,1, figsize=(3*3, 8), dpi = 120)
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(name)
        for ax in axes:
            ax.plot(T_hires, Lz_fit_trim, label = "fit", c = "darkorange", lw = 2, ls = "-")
            ax.plot(df["T"], df["Lz"], label = "All data", marker = "x", lw = 0, c = "grey", ms = 4)
            ax.plot(dftrim["T"], dftrim["Lz"], label = "Utilised data", marker = "o", lw = 0, c = "navy", ms = 4)
            
            ax.legend(loc="upper right", bbox_to_anchor=(1.25,1))
            ax.grid(alpha = 0.3)

        ax = axes[0]

        for i, label in enumerate(["Full range", "0-200eV", "0-3eV"]):
            axes[i].set_title(label, loc="right")
            if i == 2: axes[i].set_xlabel("T [eV]") 
            axes[i].set_ylabel("Lz [W/m3]")

        axes[1].set_xlim(0,200)
        axes[2].set_xlim(0,3)
        axes[2].set_yscale("log")

    if print_output is True:
        print("Hermes-3 copy-paste coefficients-------------------------")

        print(f""" 
            if (Te >= {fit_limits[0]} and Te <= {fit_limits[1]}) {{
            log_out = log_out""" )
        for i, x in enumerate(coeffs):
            if i == len(coeffs)-1:
                print(f"        {x:+.4e} * pow(logT, {i});")  # Semicolon at end
            else:
                print(f"        {x:+.4e} * pow(logT, {i})")
            
        print(f"""        return exp(log_out);""")
        print(f"""
        }} else if (Te < {fit_limits[0]}) {{
            return {Lz_lo:.4e};   
        }} else {{
            return {Lz_hi:.4e};
        }}
        """)
        
        def fit_fun(Te):
            
            # Replicate any rounding error in Hermes-3
            rounding = 10
            def scientific_round(x):
                return float(np.format_float_scientific(x, precision = rounding))
            
            logT = np.log(Te)
            log_out = 0

            if Te >= fit_limits[0] and Te <= fit_limits[1]:
                log_out = log_out          \
                +np.round(scientific_round(coeffs[0]), rounding) * pow(logT, 0) \
                +np.round(scientific_round(coeffs[1]), rounding) * pow(logT, 1) \
                +np.round(scientific_round(coeffs[2]), rounding) * pow(logT, 2) \
                +np.round(scientific_round(coeffs[3]), rounding) * pow(logT, 3) \
                +np.round(scientific_round(coeffs[4]), rounding) * pow(logT, 4) \
                +np.round(scientific_round(coeffs[5]), rounding) * pow(logT, 5) \
                +np.round(scientific_round(coeffs[6]), rounding) * pow(logT, 6) \
                +np.round(scientific_round(coeffs[7]), rounding) * pow(logT, 7) \
                +np.round(scientific_round(coeffs[8]), rounding) * pow(logT, 8) \
                +np.round(scientific_round(coeffs[9]), rounding) * pow(logT, 9) \
                +np.round(scientific_round(coeffs[10]),rounding) * pow(logT, 10) 
                return np.exp(log_out)

            elif Te < fit_limits[0]:
                return Lz_lo

            else:
                return Lz_hi
            
    return fit_fun