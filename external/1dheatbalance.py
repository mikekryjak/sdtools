# Matt Khan 2023


import xhermes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import sys

#----- Load in the shot data and input options using xHermes
# filePath = sys.argv[1]
# ds1       = xhermes.open(filePath,keep_yboundaries=True) # Load in with all guard cells
# ds1       = ds1.hermes.extract_1d_tokamak_geometry()      # Extract and normalise outputs
# ds1       = ds1.isel(pos=slice(1,-1))                     # Crop off outer guard cells, keep inners for target values
# ds1["t"]  = ds1["t"]*1e3                                  # Convert t into ms from s
# guardRep = False

#----- Load in the shot data and input options using sdtools
sys.path.append("/home/mbk513/sdtools")
from hermes3.load import *
filePath  = sys.argv[1]
ds2       = Load.case_1D(filePath).ds                     # Load, extract, normalise, and guard replace outputs
ds2       = ds2.isel(pos=slice(1,-1))                     # Crop off outer guard cells, keep inners for target values
ds2["t"]  = ds2["t"]*1e3                                  # Convert t into ms from s
guardRep = True

ds = ds2

print(np.shape(ds["pos"].values))
print(np.shape(ds["dv"].values))

def tanh(x,x0):
    return (np.exp(2.*(x-x0))-1.)/(np.exp(2.*(x-x0))+1.)

#----- Source up to x-point
sourceVolume = ds["dv"].values[1:-1]
sourceVolume[np.argmin(np.abs(ds["pos"].values-50.)):] = 0
sourceVolume = np.trapz(sourceVolume,ds["pos"].values[1:-1])

#----- Tanh Pump
pumpRegion     = (0.5*tanh(ds["pos"].values[1:-1],62.5)+0.5)*(-0.5*tanh(ds["pos"].values[1:-1],87.5)+0.5)
# pumpRegion[:np.argmin(np.abs(ds["pos"].values-0.))] = 0.
pumpVolumeTanh = np.trapz(ds["dv"].values[1:-1]*pumpRegion,ds["pos"].values[1:-1])

print(sourceVolume)
print(pumpVolumeTanh)
print(sourceVolume/pumpVolumeTanh)
sys.exit()

#----- Load in calculate sheath gammas, cross-sectional area, and elementary charge
options = BoutData(filePath, yguards = True, info = False, strict = True, DataFileCaching=False)["options"]
gamma_e = float(options["sheath_boundary_simple"]["gamma_e"])
gamma_i = float(options["sheath_boundary_simple"]["gamma_i"])
csArea  = ds["J"].values/np.sqrt(ds["g_22"].values)
q_e     = 1.602e-19

#----- Calculate the target heat flux from the inner guard cells and the final domain cell
if(guardRep):
    Q_t_electrons = gamma_e*ds["Ne"][:,-1] *q_e*ds["Te"][:,-1] *ds["Ve"][:,-1] *csArea[-1]#ds["J"]/np.sqrt(ds["g_22"])#ds["da"][0,-1]
    Q_t_ions      = gamma_i*ds["Nd+"][:,-1]*q_e*ds["Td+"][:,-1]*ds["Vd+"][:,-1]*csArea[-1]#ds["J"]/np.sqrt(ds["g_22"])#ds["da"][0,-1]
    #----- Units  = [m^-3]*[J]*[m.s^-1]*[m^2] = J.s^-1 = W
else:
    Q_t_electrons = gamma_e*(ds["Ne"][:,-1] +ds["Ne"][:,-2]) *0.5*q_e*(ds["Te"][:,-1] +ds["Te"][:,-2]) *0.5*(ds["Ve"][:,-1] +ds["Ve"][:,-2]) *0.5*csArea[-1]
    Q_t_ions      = gamma_i*(ds["Nd+"][:,-1]+ds["Nd+"][:,-2])*0.5*q_e*(ds["Td+"][:,-1]+ds["Td+"][:,-2])*0.5*(ds["Vd+"][:,-1]+ds["Vd+"][:,-2])*0.5*csArea[-1]
Q_t = Q_t_electrons+Q_t_ions

#----- Crop off inner guard cells or the boundary values as they are no longer needed
ds = ds.isel(pos=slice(1,-1))

#----- Old calculations for the total energy contained within each species
# totalElectronEng = (3./2.)*(ds["Pe"]*ds["dv"]).sum("pos")#* (ds["t"] - ds["t"][0])
# totalIonEng      = (3./2.)*(ds["Pd+"]*ds["dv"]).sum("pos")#* (ds["t"] - ds["t"][0])
# totalNeutralEng  = (3./2.)*(ds["Pd"]*ds["dv"]).sum("pos")#* (ds["t"] - ds["t"][0])
# totalSystemEng   = totalElectronEng+totalIonEng+totalNeutralEng

#----- Total Input Power
totalInputPow   = (3./2.)*((ds["Pe_src"]*ds["dv"]).sum("pos")+(ds["Pd+_src"]*ds["dv"]).sum("pos"))# Pe_src is in Pa.s^-1 which is J.m^-3.s^-1 = W.m^-3
#----- Impurity Radiation
totalRadArPow   = (ds["Rar"]*ds["dv"]).sum("pos")
#----- Hydrogenic Radiation
totalRadHexPow  = np.abs((ds["Rd+_ex"] *ds["dv"]).sum("pos")) # Needs to be negative because of output conventions
totalRadHrecPow = np.abs((ds["Rd+_rec"]*ds["dv"]).sum("pos"))
#----- CX
totalCXKinetic  = 0.5*((abs(ds["Fdd+_cx"])*abs(ds["Vd+"]))*ds["dv"]).sum("pos") # E = 0.5* mv*v ???
totalCXTherm    = (ds["Edd+_cx"]*ds["dv"]).sum("pos")
#----- Total energy needed to ionise all the neutrals
totalIoniseEng  = (ds["Nd"]*ds["dv"]*13.6*q_e).sum("pos")
timeToIonise    = totalIoniseEng.values[-1]/totalInputPow.values[-1] # J/(J/s) = s
print("For input power: %.3e , total time needed to fully ionised neutrals (for zero losses) = %.3e ms"%(totalInputPow.values[-1],timeToIonise*1e3))

#----- Total Power Loss
totalPowLoss    = Q_t+totalRadArPow+totalRadHexPow+totalRadHrecPow

plt.plot(ds["t"].values,totalInputPow.values,  label="totalInputPow")
plt.plot(ds["t"].values,totalIoniseEng.values,  label="totalIoniseEng")
plt.plot(ds["t"].values,totalRadArPow.values,  label="totalRadArPow")
plt.plot(ds["t"].values,totalRadHexPow.values, label="totalRadHexPow")
plt.plot(ds["t"].values,totalRadHrecPow.values,label="totalRadHrecPow")
plt.plot(ds["t"].values,Q_t,                   label="Q_t")
# plt.plot(ds["t"].values,Q_t_ions,              label="Q_t_ions")
# plt.plot(ds["t"].values,Q_t_electrons,         label="Q_t_electrons")
plt.plot(ds["t"].values,totalPowLoss.values,   label="totalPowLoss")
# plt.ylim([-0.1e9,1.1e9])
plt.legend(loc="best")
plt.xlabel("Time (ms)")
plt.ylabel("Power (W)")
plt.show()

# mike_heat_balance()