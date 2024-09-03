from RMK_support import RKWrapper
import RMK_support.simple_containers as sc
import RMK_support.IO_support as io
import RMK_support.dashboard_support as ds
import pickle
import os
import regex as re

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from hermes3.

def constants(name):
    
    d = dict()
    d["mass_p"] = 1.6726219e-27 # Proton mass [kg]
    d["mass_e"] = 9.1093837e-31 # Electron mass [kg]
    d["a0"] = 5.29177e-11 # Bohr radius [m]
    d["q_e"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    d["k_b"] = 1.3806488e-23 # Boltzmann self.ant [JK^-1]
    d["e0"] = 8.854187817e-12 # Vacuum permittivity [Fm^-1]
    
    return d[name]

def load_remkit(hdf5Filepath, wrapperPickleFilename, numFiles):
    
    wrapper = pickle.load(open(wrapperPickleFilename,"rb"))
    # assert type(wrapper) == RKWrapper
    varsNotInOutput=list(wrapper.varCont.dataset.keys()-set(wrapper.varsInOutput()))
    
    loadFilenames = [os.path.join(hdf5Filepath, f'ReMKiT1DVarOutput_{i}.h5') for i in range(numFiles)]
    loadedData = io.loadFromHDF5(wrapper.varCont,filepaths=loadFilenames,varsToIgnore=varsNotInOutput,isXinMeters=wrapper.grid.isLengthInMeters)
    ld = loadedData

    #There is no automatic unnormalisation, but you can get them here
    import RMK_support.sk_normalization as skn
    norms = wrapper.normalization
    n = skn.calculateNorms(norms["eVTemperature"], norms["density"], norms["referenceIonZ"])
    n = {**norms, **n}

    # Manually normalise some of the key variables

    ld["Te"] *= n["eVTemperature"]
    ld["Ti"] *= n["eVTemperature"]
    ld["Te_dual"] *= n["eVTemperature"]
    ld["Ti_dual"] *= n["eVTemperature"]
    ld["ne"] *= n["density"]
    ld["ne_dual"] *= n["density"]
    ld["Wi"] *= n["eVTemperature"]*n["density"] * constants("q_e")
    ld["We"] *= n["eVTemperature"]*n["density"] * constants("q_e")
    ld["W"] = ld["We"] + ld["Wi"]
    ld["Ge"] *= n["density"] * n["speed"]
    ld["Gi"] *= n["density"] * n["speed"]
    ld["qi"] *= n["heatFlux"] * 1e-6    # Conductive flux [MW/m2]
    ld["qe"] *= n["heatFlux"] * 1e-6    # Conductive flux [MW/m2]

    ld["Ne"] = ld["ne"]
    ld["Td+"] = ld["Ti"]
    ld["Te"] = ld["Te"]
    ld["Vd+"] = ld["ui"]
    ld["Pd+"] = ld["Ne"] * ld["Td+"] * constants("q_e")
    ld["Pe"] = ld["Ne"] * ld["Te"] * constants("q_e")
    ld["P"] = ld["Pd+"] + ld["Pe"]
    
    ld["Ne"] = ld["ne"]
    ld["Ne_dual"] = ld["ne_dual"]
    ld["Td+"] = ld["Ti"]
    ld["Td+_dual"] = ld["Ti_dual"]
    ld["Te"] = ld["Te"]
    ld["Te_dual"] = ld["Te_dual"]
    ld["Vd+"] = ld["ui"]
    ld["Vd+_dual"] = ld["ui_dual"]
    
    for param in ld:
    
        # Unnormalise heat fluxes in domain and on boundary
        if re.search(r"fluidBase|bohmBoundary_.*_energy_.*", param) is not None:
            # ld[param] *= ld["We"].attrs["normSI"] / n["time"] * 1e-6 
            
            ld[param] *= n["density"] * n["eVTemperature"] * constants("q_e") / n["time"] * 1e-6 
            
    return wrapper, ld

# def hbalance_comparison(ds, rkds):
    