import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pathlib
import xarray
from sd1d.analysis import AMJUEL
from hermes3.utils import *

class NeutralTransport():
    def __init__(self, ds):
        
        if len(ds.dims) > 1:
            raise Exception("Only 1D radial slices are supported.")
        
        self.ds = ds
        self.dist = (ds["R"] - ds["R"][ds.metadata["ixseps1"]]).values
        
    def get_rates(self, Te, Ta, Ti, Ne, Na, reproduce_bug = True):
        """
        Reproduce collisionalities from Hermes-3
        Can reproduce bugs:
        - EN had Te[i] in SI instead of normalised, resulting in K being too off by sqrt(Tnorm)
        - NI had temperature2 in normalised instead of SI, resulting in K being off by sqrt(1/Tnorm)
        """
        
        ds = self.ds
        domain = ds
        m = ds.metadata
        dist = self.dist
        q_e = constants("q_e")
        mp = constants("mass_p")
        me = constants("mass_e")

        vth_n = np.sqrt(q_e*Ta / (mp*2))
        
        # IZ/CX (tables) ----------------
        # CSV precision may affect results
        rtools = AMJUEL()
        nu_iz = np.zeros_like(Ta)
        nu_cx = np.zeros_like(Ta)
        for i, _ in enumerate(dist):
            nu_iz[i] = rtools.amjuel_2d("H.4 2.1.5", Te[i], Ne[i]) * Ne[i]
            nu_cx[i] = rtools.amjuel_1d("H.2 3.1.8", (Ta[i] + Ti[i])/2) * Ne[i]
    
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
        """
        
        mp = constants("mass_p")
        q_e = constants("q_e")
        nu = 0
        
        for c in collisions:
            nu += self.k[c]
            
        vth_n = np.sqrt(Ta*q_e / (mp*2))
            
        self.Dn = vth_n**2 / (nu * mp*2)
        
        
        