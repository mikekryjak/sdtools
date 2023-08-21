import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pathlib
import xarray
from sd1d.analysis import AMJUEL
from hermes3.utils import *

class NeutralTransport():
    def __init__(self, ds):
        
        if len(ds.dims) == 2:
            print("2D data detected")
        
        self.ds = ds
        
    def get_rates(self, Te, Ta, Ti, Ne, Na, reproduce_bug = True):
        """
        Reproduce collisionalities from Hermes-3
        Can reproduce bugs:
        - EN had Te[i] in SI instead of normalised, resulting in K being too off by sqrt(Tnorm)
        - NI had temperature2 in normalised instead of SI, resulting in K being off by sqrt(1/Tnorm)
        """
        
        ds = self.ds
        m = ds.metadata
        q_e = constants("q_e")
        mp = constants("mass_p")
        me = constants("mass_e")

        vth_n = np.sqrt(q_e*Ta / (mp*2))
        
        # IZ/CX (tables) ----------------
        # CSV precision may affect results
        rtools = AMJUEL()
        nu_iz = np.zeros_like(Ta)
        nu_cx = np.zeros_like(Ta)
        
        if len(Ne.shape) == 1:
            for i, _ in enumerate(Te):
                nu_iz[i] = rtools.amjuel_2d("H.4 2.1.5", Te[i], Ne[i]) * Ne[i]
                nu_cx[i] = rtools.amjuel_1d("H.2 3.1.8", (Ta[i] + Ti[i])/2) * Ne[i]
                
        elif len(Ne.shape) == 2:
            for i in range(Te.shape[0]):
                for j in range(Te.shape[1]):
                    nu_iz[i,j] = rtools.amjuel_2d("H.4 2.1.5", Te[i,j], Ne[i,j]) * Ne[i,j]
                    nu_cx[i,j] = rtools.amjuel_1d("H.2 3.1.8", (Ta[i,j] + Ti[i,j])/2) * Ne[i,j]
                    
        # CX (Hermes-3) ----------------
        # ln_sigmav = -18.5028
        # Teff = (Ta + Ti)/2
        # lnT = np.log(Teff)
        # lnT_n = lnT.copy()

        # for b in [0.3708409, 7.949876e-3, -6.143769e-4, -4.698969e-4, -4.096807e-4, 1.440382e-4, -1.514243e-5, 5.122435e-7]:
        #     ln_sigmav += b * lnT_n
        #     lnT_n *= lnT
            
        # sigmav_cx = np.exp(ln_sigmav) * 1e-6 # convert from cm^3/s to m^3/s
            
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
        
    def get_Dn(self, Ta,
               collisions = ["cx", "nn", "ne", "ni", "nnlim"]):
        """
        Calculate diffusion coefficient from collision rates
        Inputs: collisions is a list of strings corresponding to self.k.
        It defaults to Hermes-3 default
        
        To calculate fluxes from this, do:
        flux = Dn * Nd * np.gradient(Pd, x) / Pd / (dy * dz)
        """
        
        mp = constants("mass_p")
        q_e = constants("q_e")
        
        nu = 0
        for c in collisions:
            nu += self.k[c]
            
        vth_n = np.sqrt(Ta*q_e / (mp*2))
            
        self.Dn = vth_n**2 / (nu)

        
        
        