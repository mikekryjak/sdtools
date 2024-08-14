from boutdata import collect
import sys
import numpy as np

if len(sys.argv) != 2:
    print("Usage: {} path".format(sys.argv[0]))
    sys.exit(1)

path = sys.argv[1]


AA = 2 # Ion atomic mass number
AA_e = 1. / 1836 # Electron atomic mass number
Zi = 1 # Ion charge number
sheath_ion_polytropic = 1.0

gamma_e = 4.5
gamma_i = 2.5

Nnorm = collect("Nnorm", path=path, info=False)
Tnorm = collect("Tnorm", path=path, info=False)
rho_s0 = collect("rho_s0", path=path, info=False)
qe = 1.602e-19
Pnorm = qe * Tnorm * Nnorm
wci = collect("Omega_ci", path=path, info=False)
Cs0 = rho_s0 * wci

J = collect("J", path=path, info=False).squeeze()
dx = collect("dx", path=path, info=False).squeeze()
dy = collect("dy", path=path, info=False).squeeze()
dz = collect("dz", path=path, info=False).squeeze()

dV = J * dx * dy * dz * rho_s0**3

def sumSource(name):
    value = collect(name, path = path, tind=-1, info=False).squeeze()
    return np.sum(dV * value)

def sheathValue(name):
    value = collect(name, path = path, tind=-1, info=False, yguards=True).squeeze()
    return 0.5 * (value[-3] + value[-2])

J_sheath = sheathValue("J")
g_22 = collect("g_22", path = path, tind=-1, info=False, yguards=True).squeeze()
sqrtg_22_sheath = 0.5 * (np.sqrt(g_22[-3]) + np.sqrt(g_22[-2]))
dx_sheath = sheathValue("dx")
dz_sheath = sheathValue("dz")

total_volume = np.sum(dV)
area_sheath = J_sheath * dx_sheath * dz_sheath / sqrtg_22_sheath * rho_s0**2
print("Simulation geometry")
print(f"    Volume: {total_volume} m^3")
print(f"    Area: {area_sheath} m^2")
print(f"    Volume / area: {total_volume / area_sheath} m")
print("")


###########################################################
# Particle balance

particle_source = wci * Nnorm * sumSource("Sd+_src")
ionization = wci * Nnorm * sumSource("Sd+_iz")
recombination = wci * Nnorm * sumSource("Sd+_rec")
total_ion_source = wci * Nnorm * sumSource("SNd+")
feedback_source = wci * Nnorm * sumSource("Sd+_feedback")

print(f"Total ion particle source:  {total_ion_source}  (check: {particle_source + ionization + recombination + feedback_source}")
print(f"  |- External ion source:   {particle_source}")
print(f"  |- Feedback source:       {feedback_source}")
print(f"  |- Ionization source:     {ionization}")
print(f"  |- Recombination source:  {recombination}")
print("")

total_neutral_source = wci * Nnorm * sumSource("SNd")
neutral_recycle = wci * Nnorm * sumSource("Sd_target_recycle")
neutral_source = wci * Nnorm * sumSource("Sd_src")

print(f"Total neutral particle source: {total_neutral_source} (check: {neutral_source - ionization - recombination + neutral_recycle})")
print(f"  |- External neutral source:  {neutral_source}")
print(f"  |- Target recycling:         {neutral_recycle}")
print(f"  |- Ionization source:        {-ionization}")
print(f"  |- Recombination source:     {-recombination}")
print("")

ni_sheath = sheathValue("Nd+")
te_sheath = sheathValue("Te")
ti_sheath = sheathValue("Td+")
vi_sheath = sheathValue("Vd+")

print(f"Sheath")
print(f"  Density:     {Nnorm * ni_sheath} m^-3")
print(f"  Sound speed: {Cs0 * np.sqrt((sheath_ion_polytropic * ti_sheath + Zi * te_sheath) / AA)} m/s")
print(f"  Flow speed:  {Cs0 * vi_sheath} m/s")
print(f"  Ion sink:    {Nnorm * Cs0 * ni_sheath * vi_sheath * area_sheath} s^-1")
print(f"  Neutral recycling: {neutral_recycle} s^-1")
print("")

###########################################################

ion_heating = (3/2 * Pnorm * wci) * sumSource("Pd+_src")
electron_heating = (3/2 * Pnorm * wci) * sumSource("Pe_src")

print(f"Total input power:     {(ion_heating + electron_heating) * 1e-6} MW")
print(f"  |- Ion heating:      {ion_heating * 1e-6} MW")
print(f"  |- Electron heating: {electron_heating * 1e-6} MW")
print("")

recycle_heating = Pnorm * wci * sumSource("Ed_target_recycle")
ion_energy_flux = gamma_i * (Pnorm * Cs0) * ni_sheath * ti_sheath * vi_sheath * area_sheath
electron_energy_flux = gamma_e * (Pnorm * Cs0) * ni_sheath * te_sheath * vi_sheath * area_sheath

ion_convection = 2.5 * (Pnorm * Cs0) * ni_sheath * ti_sheath * vi_sheath * area_sheath
ion_kinetic = (Pnorm * Cs0) * 0.5 * vi_sheath**2 * AA * ni_sheath * vi_sheath * area_sheath

electron_convection = 2.5 * (Pnorm * Cs0) * ni_sheath * te_sheath * vi_sheath * area_sheath
electron_kinetic = (Pnorm * Cs0) * 0.5 * vi_sheath**2 * AA_e * ni_sheath * vi_sheath * area_sheath

R_rec = Pnorm * wci * sumSource("Rd+_rec")
R_ex = Pnorm * wci * sumSource("Rd+_ex")

print(f"Total power loss: {(ion_energy_flux + electron_energy_flux - recycle_heating - R_ex - R_rec) * 1e-6} MW")
print(f"  |- Ions:              {ion_energy_flux * 1e-6} MW")
print(f"      |- Convection          {ion_convection * 1e-6} MW")
print(f"      |- Kinetic energy      {ion_kinetic * 1e-6} MW")
print(f"      |- Conduction          {(ion_energy_flux - ion_kinetic - ion_convection) * 1e-6} MW")
print(f"  |- Electrons:         {electron_energy_flux * 1e-6} MW")
print(f"      |- Convection          {electron_convection * 1e-6} MW")
print(f"      |- Kinetic energy      {electron_kinetic * 1e-6} MW")
print(f"      |- Conduction          {(electron_energy_flux - electron_kinetic - electron_convection) * 1e-6} MW")
print(f"  |- Recycled neutrals: {-recycle_heating * 1e-6} MW")
print(f"  |- Ionization:        {-R_ex * 1e-6} MW")
print(f"  |- Recombination:     {-R_rec * 1e-6} MW")
print("")
