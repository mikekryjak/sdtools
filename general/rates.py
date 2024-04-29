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

def AMJUEL_rates(name):

    rates = {}
    rates["iz"] = np.array([
        [-32.4802533034, -0.05440669186583, 0.09048888225109, -0.04054078993576,
        0.008976513750477, -0.001060334011186, 6.846238436472e-05, -2.242955329604e-06,
        2.890437688072e-08],
        [14.2533239151, -0.0359434716076, -0.02014729121556, 0.0103977361573,
        -0.001771792153042, 0.0001237467264294, -3.130184159149e-06, -3.051994601527e-08,
        1.888148175469e-09],
        [-6.632235026785, 0.09255558353174, -0.005580210154625, -0.005902218748238,
        0.001295609806553, -0.0001056721622588, 4.646310029498e-06, -1.479612391848e-07,
        2.85225125832e-09],
        [2.059544135448, -0.07562462086943, 0.01519595967433, 0.0005803498098354,
        -0.0003527285012725, 3.201533740322e-05, -1.835196889733e-06, 9.474014343303e-08,
        -2.342505583774e-09],
        [-0.442537033141, 0.02882634019199, -0.00728577148505, 0.0004643389885987,
        1.145700685235e-06, 8.493662724988e-07, -1.001032516512e-08, -1.476839184318e-08,
        6.047700368169e-10],
        [0.06309381861496, -0.00578868653578, 0.00150738295525, -0.0001201550548662,
        6.574487543511e-06, -9.678782818849e-07, 5.176265845225e-08, 1.29155167686e-09,
        -9.685157340473e-11],
        [-0.005620091829261, 0.000632910556804, -0.0001527777697951, 8.270124691336e-06,
        3.224101773605e-08, 4.377402649057e-08, -2.622921686955e-09, -2.259663431436e-10,
        1.161438990709e-11],
        [0.0002812016578355, -3.564132950345e-05, 7.222726811078e-06, 1.433018694347e-07,
        -1.097431215601e-07, 7.789031791949e-09, -4.197728680251e-10, 3.032260338723e-11,
        -8.911076930014e-13],
        [-6.011143453374e-06, 8.089651265488e-07, -1.186212683668e-07, -2.381080756307e-08,
        6.271173694534e-09, -5.48301024493e-10, 3.064611702159e-11, -1.355903284487e-12,
        2.935080031599e-14]
        ])

    rates["cx"] = np.array([
        [-28.58858570847, 0.02068671746773, -0.007868331504755, 0.003843362133859,
        -0.0007411492158905, 9.273687892997e-05, -7.063529824805e-06, 3.026539277057e-07,
        -5.373940838104e-09],
        [-0.7676413320499, 0.0127800603259, -0.01870326896978, 0.00382855504889,
        -0.0003627770385335, 4.401007253801e-07, 1.932701779173e-06, -1.176872895577e-07,
        2.215851843121e-09],
        [0.002823851790251, -0.001907812518731, 0.01121251125171, -0.003711328186517,
        0.0006617485083301, -6.860774445002e-05, 4.508046989099e-06, -1.723423509284e-07,
        2.805361431741e-09],
        [-0.01062884273731, -0.01010719783828, 0.004208412930611, -0.00100574441054,
        0.0001013652422369, -2.044691594727e-06, -4.431181498017e-07, 3.457903389784e-08,
        -7.374639775683e-10],
        [0.001582701550903, 0.002794099401979, -0.002024796037098, 0.0006250304936976,
        -9.224891301052e-05, 7.546853961575e-06, -3.682709551169e-07, 1.035928615391e-08,
        -1.325312585168e-10],
        [-0.0001938012790522, 0.0002148453735781, 3.393285358049e-05, -3.746423753955e-05,
        7.509176112468e-06, -8.688365258514e-07, 7.144767938783e-08, -3.367897014044e-09,
        6.250111099227e-11],
        [6.041794354114e-06, -0.0001421502819671, 6.14387907608e-05, -1.232549226121e-05,
        1.394562183496e-06, -6.434833988001e-08, -2.746804724917e-09, 3.564291012995e-10,
        -8.55170819761e-12],
        [1.742316850715e-06, 1.595051038326e-05, -7.858419208668e-06, 1.774935420144e-06,
        -2.187584251561e-07, 1.327090702659e-08, -1.386720240985e-10, -1.946206688519e-11,
        5.745422385081e-13],
        [-1.384927774988e-07, -5.664673433879e-07, 2.886857762387e-07, -6.591743182569e-08,
        8.008790343319e-09, -4.805837071646e-10, 6.459706573699e-12, 5.510729582791e-13,
        -1.680871303639e-14]
        ])
    
    rates["Rrec"] = np.array([
        [-25.92450349909, 0.01222097271874, 4.278499401907e-05, 0.001943967743593,
        -0.0007123474602102, 0.0001303523395892, -1.186560752561e-05, 5.334455630031e-07,
        -9.349857887253e-09],
        [-0.7290670236493, -0.01540323930666, -0.00340609377919, 0.001532243431817,
        -0.0004658423772784, 5.972448753445e-05, -4.070843294052e-06, 1.378709880644e-07,
        -1.818079729166e-09],
        [0.02363925869096, 0.01164453346305, -0.005845209334594, 0.002854145868307,
        -0.0005077485291132, 4.211106637742e-05, -1.251436618314e-06, -1.626555745259e-08,
        1.073458810743e-09],
        [0.003645333930947, -0.001005820792983, 0.0006956352274249, -0.0009305056373739,
        0.0002584896294384, -3.294643898894e-05, 2.112924018518e-06, -6.544682842175e-08,
        7.8102930757e-10],
        [0.001594184648757, -1.582238007548e-05, 0.0004073695619272, -9.379169243859e-05,
        1.490890502214e-06, 2.245292872209e-06, -3.150901014513e-07, 1.631965635818e-08,
        -2.984093025695e-10],
        [-0.001216668033378, -0.0003503070140126, 0.0001043500296633, 9.536162767321e-06,
        -6.908681884097e-06, 8.232019008169e-07, -2.905331051259e-08, -3.169038517749e-10,
        2.442765766167e-11],
        [0.0002376115895241, 0.0001172709777146, -6.695182045674e-05, 1.18818400621e-05,
        -4.381514364966e-07, -6.936267173079e-08, 6.592249255001e-09, -1.778887958831e-10,
        1.160762106747e-12],
        [-1.930977636766e-05, -1.318401491304e-05, 8.848025453481e-06, -2.07237071139e-06,
        2.055919993599e-07, -7.489632654212e-09, -7.073797030749e-11, 1.047087505147e-11,
        -1.87744627135e-13],
        [5.599257775146e-07, 4.977823319311e-07, -3.615013823092e-07, 9.466989306497e-08,
        -1.146485227699e-08, 6.772338917155e-10, -1.776496344763e-11, 7.199195061382e-14,
        3.929300283002e-15]])
    
    return rates[name]
