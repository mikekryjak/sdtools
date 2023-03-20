import xarray as xr
from hermes3.utils import *

def calculate_fluxes(ds):   
    
    
    for name in ds.metadata["charged_species"]:
        L, R  =  Div_a_Grad_perp_upwind_fast(ds, ds[f"N{name}"] * ds[f"anomalous_Chi_{name}"], constants("q_e") * ds[f"T{name}"])
        ds[f"hf_perp_diff_L_{name}"] = L 
        ds[f"hf_perp_diff_R_{name}"] = R

        L, R  =  Div_a_Grad_perp_upwind_fast(ds, constants("q_e") * ds[f"T{name}"] * ds[f"anomalous_D_{name}"], ds[f"N{name}"])
        ds[f"hf_perp_conv_L_{name}"] = L 
        ds[f"hf_perp_conv_R_{name}"] = R

        ds[f"hf_perp_tot_L_{name}"] = ds[f"hf_perp_conv_L_{name}"] + ds[f"hf_perp_diff_L_{name}"]
        ds[f"hf_perp_tot_R_{name}"] = ds[f"hf_perp_conv_R_{name}"] + ds[f"hf_perp_diff_R_{name}"]

        L, R  =  Div_a_Grad_perp_upwind_fast(ds, ds[f"anomalous_D_{name}"] * ds[f"N{name}"] / ds[f"N{name}"], ds[f"N{name}"])
        ds[f"pf_perp_diff_L_{name}"] = L 
        ds[f"pf_perp_diff_R_{name}"] = R
        
        
    return ds
    

def sheath_boundary_simple(bd, gamma_e, gamma_i, Ne, Te, Ti, Zi=1, AA=1, sheath_ion_polytropic=1.0,
                           include_convective=True):
    """
    Calculate the electron and ion heat flux at the sheath, using the formula used in the
    sheath_boundary_simple component, assuming a single ion species
    with charge Zi (hydrogen=1) and atomic mass AA (hydrogen=1)

    # Returns

    flux_down, flux_up

    With units of Watts, i.e the power flowing out of each cell

    Slices at lower Y and upper Y respectively, giving heat conduction through sheath.
    Note: These do *not* include the convective heat flux, since that would usually
    be calculated in the pressure evolution (evolve_pressure component).
    """

    if not include_convective:
        gamma_e = gamma_e - 2.5
        gamma_i = gamma_i - 3.5
    
    J = bd['J']
    dx = bd['dx']
    dy = bd['dy']
    dz = bd['dz']
    g_22 = bd['g_22']

    # Lower y
    if yboundaries:
        y = 2 # First point in the domain
        ym = y - 1
        Ne_m = Ne.isel(theta=ym)
        Te_m = Te.isel(theta=ym)
        Ti_m = Ti.isel(theta=ym)
    else:
        y = 0
        ym = y # Same metric tensor component in boundary cells as in domain
        yp = y + 1 # For extrapolating boundary
        
        Ne_m = Ne.isel(theta=y)**2 / Ne.isel(theta=yp)
        Te_m = Te.isel(theta=y)**2 / Te.isel(theta=yp)
        Ti_m = Ti.isel(theta=y)**2 / Ti.isel(theta=yp)

    nesheath = 0.5 * (Ne.isel(theta=y) + Ne_m)
    tesheath = 0.5 * (Te.isel(theta=y) + Te_m)
    tisheath = 0.5 * (Ti.isel(theta=y) + Ti_m)

    qe = 1.602e-19 # Elementary charge [C]
    mp = 1.67e-27 # Proton mass [kg]
    me = 9.11e-31 # Electron mass [kg]
    
    # Ion flow speed
    C_i = np.sqrt((sheath_ion_polytropic * qe * tisheath + Zi * qe * tesheath) / (AA * mp))

    vesheath = C_i  # Assuming no current

    # Parallel heat flux in W/m^2.
    # Note: Corrected for 5/2Pe convective thermal flux, and small electron kinetic energy flux
    # so gamma_e is the total *energy* flux coefficient.
    q_e = ((gamma_e - 2.5) * qe * tesheath - 0.5 * me * vesheath**2) * nesheath * vesheath
    q_i = gamma_i * qe * tisheath * nesheath * vesheath

    # Multiply by cell area to get power
    flux_down_e = q_e * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=ym)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=ym)))
    flux_down_i = q_i * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=ym)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=ym)))

    ions_down = nesheath * vesheath * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=ym)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=ym)))
    
    # Repeat for upper Y boundary
    if yboundaries:
        y = -3 # First point in the domain
        yp = y + 1
        Ne_p = Ne.isel(theta=yp)
        Te_p = Te.isel(theta=yp)
        Ti_p = Ti.isel(theta=yp)
    else:
        y = -1
        yp = y # Same metric tensor component in boundary cells as in domain
        ym = y - 1 # For extrapolating boundary
        
        Ne_p = Ne.isel(theta=y)**2 / Ne.isel(theta=ym)
        Te_p = Te.isel(theta=y)**2 / Te.isel(theta=ym)
        Ti_p = Ti.isel(theta=y)**2 / Ti.isel(theta=ym)

    nesheath = 0.5 * (Ne.isel(theta=y) + Ne_p)
    tesheath = 0.5 * (Te.isel(theta=y) + Te_p)
    tisheath = 0.5 * (Ti.isel(theta=y) + Ti_p)
    C_i = np.sqrt((sheath_ion_polytropic * qe * tisheath + Zi * qe * tesheath) / (AA * mp))
    vesheath = C_i
    
    q_e = (gamma_e * qe * tesheath - 0.5 * me * vesheath**2) * nesheath * vesheath
    q_i = gamma_i * qe * tisheath * nesheath * vesheath

    flux_up_e = q_e * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yp)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yp)))
    flux_up_i = q_i * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yp)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yp)))

    ions_up = nesheath * vesheath * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yp)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yp)))

    return flux_down_e, flux_up_e, flux_down_i, flux_up_i, ions_down, ions_up
    
def sheath_boundary_simple_original(bd, gamma_e, gamma_i, Ne, Te, Ti, Zi=1, AA=1, sheath_ion_polytropic=1.0,
                           include_convective=True):
    """
    Calculate the electron and ion heat flux at the sheath, using the formula used in the
    sheath_boundary_simple component, assuming a single ion species
    with charge Zi (hydrogen=1) and atomic mass AA (hydrogen=1)

    # Returns

    flux_down, flux_up

    With units of Watts, i.e the power flowing out of each cell

    Slices at lower Y and upper Y respectively, giving heat conduction through sheath.
    Note: These do *not* include the convective heat flux, since that would usually
    be calculated in the pressure evolution (evolve_pressure component).
    """

    if not include_convective:
        gamma_e = gamma_e - 2.5
        gamma_i = gamma_i - 3.5
    
    J = bd['J']
    dx = bd['dx']
    dy = bd['dy']
    dz = bd['dz']
    g_22 = bd['g_22']

    # Lower y
    if yboundaries:
        y = 2 # First point in the domain
        ym = y - 1
        Ne_m = Ne.isel(theta=ym)
        Te_m = Te.isel(theta=ym)
        Ti_m = Ti.isel(theta=ym)
    else:
        y = 0
        ym = y # Same metric tensor component in boundary cells as in domain
        yp = y + 1 # For extrapolating boundary
        
        Ne_m = Ne.isel(theta=y)**2 / Ne.isel(theta=yp)
        Te_m = Te.isel(theta=y)**2 / Te.isel(theta=yp)
        Ti_m = Ti.isel(theta=y)**2 / Ti.isel(theta=yp)

    nesheath = 0.5 * (Ne.isel(theta=y) + Ne_m)
    tesheath = 0.5 * (Te.isel(theta=y) + Te_m)
    tisheath = 0.5 * (Ti.isel(theta=y) + Ti_m)

    qe = 1.602e-19 # Elementary charge [C]
    mp = 1.67e-27 # Proton mass [kg]
    me = 9.11e-31 # Electron mass [kg]
    
    # Ion flow speed
    C_i = np.sqrt((sheath_ion_polytropic * qe * tisheath + Zi * qe * tesheath) / (AA * mp))

    vesheath = C_i  # Assuming no current

    # Parallel heat flux in W/m^2.
    # Note: Corrected for 5/2Pe convective thermal flux, and small electron kinetic energy flux
    # so gamma_e is the total *energy* flux coefficient.
    q_e = ((gamma_e - 2.5) * qe * tesheath - 0.5 * me * vesheath**2) * nesheath * vesheath
    q_i = gamma_i * qe * tisheath * nesheath * vesheath

    # Multiply by cell area to get power
    flux_down_e = q_e * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=ym)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=ym)))
    flux_down_i = q_i * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=ym)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=ym)))

    ions_down = nesheath * vesheath * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=ym)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=ym)))
    
    # Repeat for upper Y boundary
    if yboundaries:
        y = -3 # First point in the domain
        yp = y + 1
        Ne_p = Ne.isel(theta=yp)
        Te_p = Te.isel(theta=yp)
        Ti_p = Ti.isel(theta=yp)
    else:
        y = -1
        yp = y # Same metric tensor component in boundary cells as in domain
        ym = y - 1 # For extrapolating boundary
        
        Ne_p = Ne.isel(theta=y)**2 / Ne.isel(theta=ym)
        Te_p = Te.isel(theta=y)**2 / Te.isel(theta=ym)
        Ti_p = Ti.isel(theta=y)**2 / Ti.isel(theta=ym)

    nesheath = 0.5 * (Ne.isel(theta=y) + Ne_p)
    tesheath = 0.5 * (Te.isel(theta=y) + Te_p)
    tisheath = 0.5 * (Ti.isel(theta=y) + Ti_p)
    C_i = np.sqrt((sheath_ion_polytropic * qe * tisheath + Zi * qe * tesheath) / (AA * mp))
    vesheath = C_i
    
    q_e = (gamma_e * qe * tesheath - 0.5 * me * vesheath**2) * nesheath * vesheath
    q_i = gamma_i * qe * tisheath * nesheath * vesheath

    flux_up_e = q_e * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yp)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yp)))
    flux_up_i = q_i * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yp)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yp)))

    ions_up = nesheath * vesheath * dx.isel(theta=y) * dz.isel(theta=y) * (J.isel(theta=y) + J.isel(theta=yp)) / (np.sqrt(g_22.isel(theta=y)) + np.sqrt(g_22.isel(theta=yp)))

    return flux_down_e, flux_up_e, flux_down_i, flux_up_i, ions_down, ions_up

def Div_a_Grad_perp_upwind_fast(ds, a, f):
    """
    AUTHOR: M KRYJAK 15/03/2023
    Rewrite of Div_a_Grad_perp_upwind but using fast Xarray functions
    Runs very fast, but cannot be easily verified against the code
    Full comments available in the slower version
    
    Parameters
    ----------
    bd : xarray.Dataset
        Dataset with the simulation results for geometrical info
    a : np.array
        First term in the derivative equation, e.g. chi * N
    f : np.array
        Second term in the derivative equation, e.g. q_e * T
    
    Returns
    ----------
    Tuple of two quantities:
    (F_L, F_R)
    These are the flow into the cell from the left (x-1),
    and the flow out of the cell to the right (x+1). 
    NOTE: *NOT* the flux; these are already integrated over cell boundaries

    Example
    ----------
    F_L, F_R = Div_a_Grad_perp_upwind(ds, chi * N, e * T)
    - chi is the heat diffusion in m^2/2
    - N is density in m^-3
    - T is temperature in eV
    Then F_L and F_R would have units of Watts,
    The time-derivative of the pressure in that cell would be:
    d/dt (3/2 P) = (F_L - F_R) / V
    where V = dx * dy * dz * J is the volume of the cell
    
    """
    
    J = ds["J"] # Jacobian
    g11 = ds["g11"]
    dx = ds["dx"]
    dy = ds["dy"]
    dz = ds["dz"]
    
    # shift(x=-1) returns array of x[i+1]s because it moves the array 1 step towards start
    # So this is equivalent to (f[i+1] - f[i]) * (J[i]*g11[i] - J[i+1]*g11[i+1]) / (dx[i] + dx[i+1])

    gradient = (f.shift(x=-1) - f) * (J*g11 + J.shift(x=-1)*g11.shift(x=-1)) / (dx + dx.shift(x=-1))
    flux = -gradient * 0.5 * (a + a.shift(x=-1))
    flux *= dy * dz
    
    F_R = flux
    F_L = flux.shift(x=1)  # the shift of 1 index to get F_L because the left flux at cell X is the same as the right flux at cell X-1.
    
    return F_L, F_R


def Div_a_Grad_perp_upwind(bd, a, f):
    """
    AUTHOR: B DUDSON 2023
    Reproduction of the same function within Hermes-3 
    Used to calculate perpendicular fluxes
    Runs very slow, but can be easily verified against the code
    
    # Returns

    Tuple of two quantities:
    (F_L, F_R)

    These are the flow into the cell from the left (x-1),
    and the flow out of the cell to the right (x+1). 

    Note: *NOT* the flux; these are already integrated over cell boundaries

    # Example

    F_L, F_R = Div_a_Grad_perp_upwind(chi * N, e * T)

    - chi is the heat diffusion in m^2/2
    - N is density in m^-3
    - T is temperature in eV

    Then F_L and F_R would have units of Watts,

    The time-derivative of the pressure in that cell would be:

    d/dt (3/2 P) = (F_L - F_R) / V

    where V = dx * dy * dz * J is the volume of the cell
    
    """

    J = bd["J"] # Jacobian
    g11 = bd["g11"]
    dx = bd["dx"]
    dy = bd["dy"]
    dz = bd["dz"]

    F_R = xr.zeros_like(f) # Flux to the right
    F_L = xr.zeros_like(f) # Flux from the left

    for x in bd.x[:-1]:
        xp = x + 1  # The next X cell
        # Note: Order of array operations matters for shape of the result
        gradient = (f.isel(x=xp) - f.isel(x=x)) * (J.isel(x=x) * g11.isel(x=x) + J.isel(x=xp) * g11.isel(x=xp))  / (dx.isel(x=x) + dx.isel(x=xp))

        flux = -gradient * 0.5*(a.isel(x=x) + a.isel(x=xp))
        
        # if gradient > 0:
        #     # Flow from x+1 to x
        #     flux = -gradient * a.isel(x=xp)  # Note: Negative flux = flow into this cell from right
        # else:
        #     # Flow from x to x+1
        #     flux = -gradient * a.isel(x=x)  # Positive flux => Flow from this cell to the right

        # Need to multiply by dy * dz because these are assumed constant in X in the calculation
        # of flux and cell volume.
        flux *= dy.isel(x=x) * dz.isel(x=x)

        F_R[dict(x=x)] = flux
        F_L[dict(x=xp)] = flux

    return F_L, F_R