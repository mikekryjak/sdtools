import regex as re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys
from .utils import constants

def softFloor(arr, min):
    
    for i, value in enumerate(arr):
        arr[i] = value + min * np.exp(-value / min)

    return arr

def floor(value, min):
    if value < min:
        return min
    else:
        return value


def get_Kei_coll(ds):
    """
    Calculate Ked+_coll with hardcoded species.
    Needs a dataset with one time slice.
    """
    
    if "t" in ds.sizes:
        raise ValueError("Dataset must have only one time slice for this function to work correctly.")

    Ai = 2
    Mp = constants("mass_p")
    Me = constants("mass_e")
    qe = constants("q_e")
    e0 = constants("e0")
    Te = ds["Te"].values
    Ti = ds["Td+"].values
    Ne = ds["Ne"].values
    Ni = ds["Nd+"].values
    Zi = 1
    ei_multiplier = 1

    me_mi = Me / (Mp * Ai)

    coulomb_log = np.zeros_like(Te)

    for i in range(len(Te)):
        if Te[i] < 0.1 or Ni[i] < 1e10 or Ne[i] < 1e10:
            coulomb_log[i] = 10.0
        elif Te[i] < Ti[i] * me_mi:
            coulomb_log[i] = (
                23.0
                - 0.5 * np.log(Ni[i])
                + 1.5 * np.log(Ti[i])
                - np.log(Zi**2 * Ai)
            )
        elif Te[i] < np.exp(2) * Zi**2:
            coulomb_log[i] = (
                30.0
                - 0.5 * np.log(Ne[i])
                - np.log(Zi)
                + 1.5 * np.log(Te[i])
            )
        else:
            coulomb_log[i] = (
                31.0
                - 0.5 * np.log(Ne[i])
                + np.log(Te[i])
            )
            
    vesq = 2 * softFloor(Te, 0.1) * qe / Me;
    visq = 2 * softFloor(Ti, 0.1) * qe / (Mp * Ai);

    nu = (qe**2 * Zi)**2 * np.clip(Ni, a_min=0.0, a_max=None) * softFloor(coulomb_log, 1.0) * (1. + me_mi) / (3 * (np.pi * (vesq + visq))**1.5 * (e0 * Me)**2) * ei_multiplier
    
    return nu

def get_Kee_coll(ds):
    
    Ai = 2
    Mp = constants("mass_p")
    Me = constants("mass_e")
    qe = constants("q_e")
    e0 = constants("e0")
        
    Te = ds["Te"].values
    Ne = ds["Ne"].values
    Telim = np.clip(Te, 0.1, None)
    Nelim = np.clip(Ne, 1e10, None)
    logTe = np.log(Telim)

    coulomb_log = 30.4 - 0.5 * np.log(Nelim) + (5 / 4) * logTe - np.sqrt(1e-5 + (logTe - 2)**2 / 16)

    v1sq = 2 * Telim * qe / Me;

    denom = (3 * (np.pi * 2 * v1sq)**1.5) * (e0 * Me)**2

    nu = (qe**4) * np.clip(Ne, 0, None) * np.clip(coulomb_log, 1, None) * 2 / denom
    return nu