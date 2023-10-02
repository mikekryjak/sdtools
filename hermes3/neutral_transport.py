import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pathlib
import xarray
from sd1d.analysis import AMJUEL
from hermes3.utils import *
from hermes3.fluxes import *

class NeutralTransport():
    def __init__(self, ds, inputs):
        
        if len(ds.dims) == 2:
            print("2D data detected")
        
        self.ds = ds
        
        self.inputs = inputs
        
    def get_rates(self, reproduce_bug = True):
        """
        Reproduce collisionalities from Hermes-3
        Can reproduce bugs:
        - EN had Te[i] in SI instead of normalised, resulting in K being too off by sqrt(Tnorm)
        - NI had temperature2 in normalised instead of SI, resulting in K being off by sqrt(1/Tnorm)
        
        Input deck as a dict with:
        {Te, Ta, Ti, Ne, Na, Pa, dl, dr}
        
        """
        Te = self.inputs["Te"]
        Ta = self.inputs["Ta"]
        Ti = self.inputs["Ti"]
        Ne = self.inputs["Ne"]
        Na = self.inputs["Na"]
        
                
        ds = self.ds
        m = ds.metadata
        q_e = constants("q_e")
        mp = constants("mass_p")
        me = constants("mass_e")

        vth_n = np.sqrt(q_e*Ta / (mp*2))
        
        # IZ/CX (tables) ----------------
        # CSV precision may affect results
        # rtools = AMJUEL()
        nu_iz = np.zeros_like(Ta)
        # nu_cx = np.zeros_like(Ta)
        
        # if len(Ne.shape) == 1:
        #     for i, _ in enumerate(Te):
        #         nu_iz[i] = rtools.amjuel_2d("H.4 2.1.5", Te[i], Ne[i]) * Ne[i]
        #         nu_cx[i] = rtools.amjuel_1d("H.2 3.1.8", (Ta[i] + Ti[i])/2) * Ne[i]
                
        # elif len(Ne.shape) == 2:
        #     for i in range(Te.shape[0]):
        #         for j in range(Te.shape[1]):
        #             nu_iz[i,j] = rtools.amjuel_2d("H.4 2.1.5", Te[i,j], Ne[i,j]) * Ne[i,j]
        #             nu_cx[i,j] = rtools.amjuel_1d("H.2 3.1.8", (Ta[i,j] + Ti[i,j])/2) * Ne[i,j]
                    
        # CX (Hermes-3) ----------------
        ln_sigmav = -18.5028
        Teff = (Ta + Ti)/2
        lnT = np.log(Teff)
        lnT_n = lnT.copy()

        for b in [0.3708409, 7.949876e-3, -6.143769e-4, -4.698969e-4, -4.096807e-4, 1.440382e-4, -1.514243e-5, 5.122435e-7]:
            ln_sigmav += b * lnT_n
            lnT_n *= lnT
            
        nu_cx = np.exp(ln_sigmav) * 1e-6 * Ne # convert from cm^3/s to m^3/s
            
        # NN ----------------
        # Neutral-neutral collisions... not verified yet 
        a0 = np.pi * (2.8e-10)**2   # AKA 2.5e-19
        vrel = np.sqrt(2 * q_e * (Ta/(mp*2) + Ta/(mp*2)))
        nu_nn = vrel * Na *  a0 /2
        
        # EN ----------------
        # Electron-neutral collisions
        a0 = 5e-19
        vth_e = np.sqrt( (Te*q_e / me))     # In Hermes-3 Me is normalised by Mp
        if reproduce_bug is True:
            vth_e *= np.sqrt(m["Tnorm"]) ## RECONSTRUCTION OF MISTAKE IN HERMES-3 WHERE Te WAS NOT NORMALISED
        nu_en = vth_e * Na * a0
        nu_ne = nu_en * me/(mp*2) * Ne/ Na
        
        # NI ----------------
        # Neutral-ion collisions (note this is already counted in CX rate so it's double counting)
        a0 = 5e-19 
        if reproduce_bug is True:
            vrel = np.sqrt((q_e * Ta / (mp*2)) + (q_e * Ti/m["Tnorm"] / (mp*2)) )   # REPRODUCING HERMES-3 BUG
        else:
            vrel = np.sqrt((q_e * Ta / (mp*2)) + (q_e * Ti / (mp*2)) )   
        nu_ni = vrel * Ne * a0
        nu_in = nu_ni * Na / Ne
        
        # NNlim ----------------   
        # Pseudo-rate used as filler if no other collisions and as a pseudo-flux limiter
        nu_nnlim = vth_n / 0.1   # Forced to 0.1m no matter what
        
        self.k = {
            "cx" : nu_cx,
            "iz" : nu_iz,
            "nn" : nu_nn,
            "en" : nu_en,
            "ne" : nu_ne,
            "ni" : nu_ni,
            "nnlim" : nu_nnlim,
        }
        
    def get_Dn(self, 
               collisions = ["cx", "nn", "ne", "ni", "nnlim"],
               flux_limiter = "original",
               alpha = 1):
        """
        Calculate diffusion coefficient from collision rates
        Inputs: collisions is a list of strings corresponding to self.k.
        It defaults to Hermes-3 default
        
        To calculate fluxes from this, do:
        flux = Dn * Nd * np.gradient(Pd, x) / Pd / (dy * dz)
        
        You need dl. It's as below. Ensure metrics are normalised if you use them
        dr = dx/(R*Bpxy) = dx/sqrt(g11)
        dl = dy/hthe
        
        Inputs
        ------
        collisions : list of strings
            List of collisions to include in calculation
        flux_limiter : string  
            Flux limiter to use. Options are "original" and "none"
        alpha : float
            Flux limiter parameter. Default is 1.0
        """
        
        Ta = self.inputs["Ta"]
        Na = self.inputs["Na"]
        Pa = self.inputs["Pa"]
        dl = self.inputs["dl"]
        dr = self.inputs["dr"]
        mp = constants("mass_p")
        q_e = constants("q_e")
        
        nu = 0
        for c in collisions:
            nu += self.k[c]
            
        vth_n = np.sqrt(Ta*q_e / (mp*2))
            
        Dn = vth_n**2 / (nu)
        
        # Calculate Grad(ln(Pa))
        # See docstring at the end for how I got this
        nx = Ta.shape[0]
        ny = Ta.shape[1]
        dfdx = np.zeros_like(Na)
        dfdy = np.zeros_like(Na)
        f = np.log(Pa)
        if flux_limiter == "original":
            
            for i in range(nx):
                for j in range(ny):
                    if i > 0 and i < nx-1 and j > 0 and j < ny-1:
                        dfdx[i,j] = (f[i+1,j] - f[i-1,j]) / (2*dr[i,j])
                        dfdy[i,j] = (f[i,j+1] - f[i,j-1]) / (2*dl[i,j])
            
            grad_lnP = np.sqrt(dfdx**2 + dfdy**2)
            Dmax = alpha * vth_n / (grad_lnP + 1/0.1)   # The 1/0.1 is the minimum MFP of 0.1 metres

            self.Dmax = Dmax
            self.Dn_unlim = Dn.copy()
            Dn = np.where(Dn > Dmax, Dmax, Dn)
            self.Dn = Dn
            
        else:
            self.Dn = Dn
            
        
    def get_flux(self):
        
        Dn = self.Dn 
        Na = self.inputs["Na"]
        Pa = self.inputs["Pa"]
        dy = self.inputs["dy"]
        dz = self.inputs["dz"]
        dl = self.inputs["dl"]
        dr = self.inputs["dr"]
        J = self.inputs["J"]
        g11 = self.inputs["g11"]
        dx = self.inputs["dx"]
        ds = self.ds
        
        
        # L, R = Div_a_Grad_perp_fast(ds, Dn*Na, np.log(Pa))
        
        L,R = Div_a_Grad_perp_fast(ds, ds[f"Dn_calc"]*ds[f"Nd"], np.log(ds["Pd"]))
        
        self.flux = (L+R)/2 / (dy*dz)
        
        a = Dn * Na
        f = np.log(Pa)
        L = np.zeros_like(a)
        R = np.zeros_like(a)
        
        def get_flux_on_line(a, f, J, g11, dx, dy, dz):
            """
            Works for single radial slice
            """
            L = np.zeros_like(a)
            R = np.zeros_like(a)
            for i, _ in enumerate(a[:-1]):
                ip = i + 1  # The next cell
                
                gradient = (f[ip] - f[i]) * (J[i] * g11[i] + J[ip] * g11[ip]) / (dx[i] + dx[ip])
                flux = -gradient * 0.5 * (a[i] + a[ip])
                
                flux *= dy[i] * dz[i]  
                R[i] = flux
                L[ip] = flux  
            
            return L, R
        
        if len(a.shape) > 1:
            # 2D
            for yi in range(Na.shape[1]):
                L[:,yi], R[:,yi] = get_flux_on_line(a[:,yi], f[:,yi], J[:,yi], g11[:,yi], dx[:,yi], dy[:,yi], dz[:,yi])
        else:
            # 1D
            L, R = get_flux_on_line(a, f, J, g11, dx, dy, dz)
            
        self.flux = (L+R)/2
        
        
def get_cx_rate(ds):
    """
    Calculate the CX rate using coeffs and expressions from Hermes-3
    and add them to the dataset.
    You must provide a dataset with 3 dimensions: t, x, theta
    """

    Ta = ds["Td"].values
    Ti = ds["Td+"].values
    Ne = ds["Ne"].values

    ln_sigmav = -18.5028
    Teff = (Ta + Ti)/2
    lnT = np.log(Teff)
    lnT_n = lnT.copy()

    for b in [0.3708409, 7.949876e-3, -6.143769e-4, -4.698969e-4, -4.096807e-4, 1.440382e-4, -1.514243e-5, 5.122435e-7]:
        ln_sigmav += b * lnT_n
        lnT_n *= lnT
        
    nu_cx = np.exp(ln_sigmav) * 1e-6 * Ne # convert from cm^3/s to m^3/s


    ds["K_cx"] = (["t", "x", "theta"], nu_cx)
    ds["K_cx"].attrs["metadata"] = ds.metadata
    
    return ds
        
def calculate_neutral_mfp(ds):
    Ta = ds["Td"].values
    Ti = ds["Td+"].values
    Ne = ds["Ne"].values
    Na = ds["Nd"].values
    Vi = ds["Vd+"]
    Va = ds["NVd"] / (ds["Nd"] * constants("mass_p")*2)

    ln_sigmav = -18.5028
    Teff = (Ta + Ti)/2
    lnT = np.log(Teff)
    lnT_n = lnT.copy()

    for b in [0.3708409, 7.949876e-3, -6.143769e-4, -4.698969e-4, -4.096807e-4, 1.440382e-4, -1.514243e-5, 5.122435e-7]:
        ln_sigmav += b * lnT_n
        lnT_n *= lnT
        
    nu_cx = np.exp(ln_sigmav) * 1e-6 * Ne # convert from cm^3/s to m^3/s
    nu = nu_cx  # Only consider CX for now
    v_thi = np.sqrt((ds["Td+"]*constants("q_e")) / (constants("mass_p")*2))
    v_thn = np.sqrt((ds["Td"]*constants("q_e")) / (constants("mass_p")*2))

    Vi_pol = Vi / (ds["J"] / np.sqrt(ds["g_22"]))
    Va_pol = Va / (ds["J"] / np.sqrt(ds["g_22"]))

    L_T_rad = ds["Td+"].bout.ddx()
    L_T_pol = ds["Td+"].bout.ddy()

    mfp_rad = (v_thi + v_thn) / (nu)
    mfp_pol = abs((Vi_pol - Va_pol)) / (nu)

    Kn_rad = abs(mfp_rad / L_T_rad)
    Kn_pol = abs(mfp_pol / L_T_pol)
    
    fig, axes = plt.subplots(1,2, figsize = (7,5), dpi = 150)
    Kn_rad.hermesm.clean_guards().bout.polygon(ax = axes[0], cmap = "Spectral_r", vmin = 1e-4, vmax = 1, logscale = True, add_colorbar = True)
    axes[0].set_title(r"Radial: $v_{rel}=v_{th}^{i} + v_{th}^{n}$")
    Kn_pol.hermesm.clean_guards().bout.polygon(ax = axes[1], cmap = "Spectral_r", vmin = 1e-4, vmax = 1, logscale = True, add_colorbar = True)
    axes[1].set_title(r"Poloidal: $v_{rel}=|v_{\theta}^{i} - v_{\theta}^{n}|$")
    fig.suptitle(r"Neutral Knudsen number for conduction $K_{n} = \frac{mfp}{L_{T}}$")
    fig.tight_layout()

    fig, axes = plt.subplots(1,2, figsize = (7,5), dpi = 150)
    mfp_rad.hermesm.clean_guards().bout.polygon(ax = axes[0], cmap = "Spectral_r", vmin = 1e-2, vmax = 10, logscale = True, add_colorbar = True)
    axes[0].set_title("$mfp_{r}$")
    mfp_pol.hermesm.clean_guards().bout.polygon(ax = axes[1], cmap = "Spectral_r", vmin = 1e-2, vmax = 10, logscale = True, add_colorbar = True)
    axes[1].set_title(r"$mfp_{\theta}$")
    fig.suptitle("Neutral mean free path")
    fig.tight_layout()
    
    v_the = np.sqrt((ds["Te"]*constants("q_e")) / (constants("mass_e")))
    mfp_e = v_the / (ds["Kee_coll"] + ds["Ked+_coll"])
    Kn_e = mfp_e / np.sqrt(abs(L_T_rad)**2 + abs(L_T_pol)**2)

    fig, axes = plt.subplots(1,2, figsize = (7,5), dpi = 150)

    mfp_e.hermesm.clean_guards().bout.polygon(ax = axes[0], cmap = "Spectral_r", vmin = None, vmax = None, logscale = True, add_colorbar = True)
    axes[0].set_title(r"Mean free path")

    Kn_e.hermesm.clean_guards().bout.polygon(ax = axes[1], cmap = "Spectral_r", vmin = None, vmax = None, logscale = True, add_colorbar = True)
    axes[1].set_title(r"Knudsen number")
    fig.suptitle(r"Electron Knudsen number for conduction $K_{n,e} = \frac{mfp}{L_{T}}$")
    fig.tight_layout()


    

"""

Hey Mike,



The Grad() operator is here:

https://github.com/boutproject/BOUT-dev/blob/master/src/field/vecops.cxx#L70



It will take derivatives in X, Y and Z directions, by default just using central differencing. It returns a covariant vector with components (x,y,z) = (DDX(ln(p)), DDY(ln(p)), DDZ(ln(p)))



The DDX, DDY, DDZ etc operators are declared here:

https://github.com/boutproject/BOUT-dev/blob/master/include/bout/coordinates.hxx#L158

and implemented here:

https://github.com/boutproject/BOUT-dev/blob/master/src/mesh/coordinates.cxx#L1562



The actual index derivatives are implemented in this file:

https://github.com/boutproject/BOUT-dev/blob/master/src/mesh/index_derivs.cxx#L74

It uses a moderately dangerous amount of template magic to generate all the possible combinations of directions, operator orders, staggered input and output etc.



The upshot of all this is that DDX(f) at (i,j,k) is just ( f(i+1, j, k) – f(i-1,j,k) ) / (2*dx(i,j,k))

Similarly, DDY(f) = ( f(i, j+1, k) – f(i,j-1,k) ) / (2*dy(i,j,k)) and DDZ(f) = ( f(i, j, k+1) – f(i,j,k-1) ) / (2*dz(i,j,k)).



The abs() function is here:

https://github.com/boutproject/BOUT-dev/blob/master/src/field/vector3d.cxx#L599

and the multiplication (dot product) here:

https://github.com/boutproject/BOUT-dev/blob/master/src/field/vector3d.cxx#L469

Because the Grad() operator returns a covariant vector we have this calculation:

https://github.com/boutproject/BOUT-dev/blob/master/src/field/vector3d.cxx#L484C19-L484C19



In your case the Z component is zero, and X-Y orthogonal so g12=0 and the calculation becomes:



f = ln(p)

gx = ( f(i+1, j, k) – f(i-1,j,k) ) / (2*dx(i,j,k))    # Grad(ln(p)).x

gy = ( f(i, j+1, k) – f(i,j-1,k) ) / (2*dy(i,j,k))    # Grad(ln(p)).y



result = g11 * gx * gx + g22 * gy * gy



Hopefully this works…



Best wishes,

Ben

"""