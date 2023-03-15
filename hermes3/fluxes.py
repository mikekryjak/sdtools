import xarray as xr

def Div_a_Grad_perp_upwind_fast(ds, a, f):
    """
    AUTHOR: M KRYJAK 15/03/2023
    Rewrite of Div_a_Grad_perp_upwind but using fast Xarray functions
    Runs very fast, but cannot be easily verified against the code
    
    Parameters
    ----------
    bd : xarray.Dataset
        Dataset with the simulation results for geometrical info
    a : np.array
        First term in the derivative equation, e.g. chi * N
    b : np.array
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
    F_R = flux
    F_L = flux.shift(x=1)  # Not sure why it needs shifting by +1 here, but this matches full function.
    
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