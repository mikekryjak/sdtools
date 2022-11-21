from boutdata.data import BoutData
import matplotlib.pyplot as plt

import numpy as np

"""
Perform a total (system, so ions and neutrals) mass balance 
for Hermes-3. This is done from scratch and only requires BoutData.
TODO: Figure out if nloss exists in Hermes-3 and include it
"""

# Collect all guard cells, so 2 on each side. The outer ones are not used. 
# Final domain grid centre is therefore index [-3].
# J/sqrt(g_22) = cross-sectional area of cell

boutdata = BoutData('/ssd_scratch/cases/htest/recycling1d', yguards = True, info = False, strict = True, DataFileCaching=False)
d = boutdata["outputs"]
o = boutdata["options"]
tind = -1
J = d["J"].squeeze()
dy = d["dy"].squeeze()
dV = J * dy
g_22 = d["g_22"].squeeze()

# Reconstruct grid position from dy
n = len(dy)
pos = np.zeros(n)
pos[0] = -0.5*dy[1]
pos[1] = 0.5*dy[1]

for i in range(2,n):
    pos[i] = pos[i-1] + 0.5*dy[i-1] + 0.5*dy[i]

def get_boundary_time(x):
    return (x[:,-2] + x[:,-3])/2

def get_boundary(x):
    return (x[-2] + x[-3])/2

# ----- Recycling
recycle_multiplier = float(o["d+"]["recycle_multiplier"])

# ----- Boundary flux
sheath_area = get_boundary(J) / np.sqrt(get_boundary(g_22))
sheath_ion_flux = get_boundary_time(d["NVd+"].squeeze())
sheath_ion_flux *= sheath_area * d["Cs0"] * d["Nnorm"]

sheath_neutral_flux = get_boundary_time(d["NVd"].squeeze())
sheath_neutral_flux *= sheath_area * d["Cs0"] * d["Nnorm"]

intended_recycle_flux = sheath_ion_flux * recycle_multiplier

# ----- Density input source
input_source = d["Sd+_src"].squeeze()[:,2:-2]
input_source = np.trapz(x = pos[2:-2], y = input_source * J[2:-2] * d["Cs0"] * d["Nnorm"])

# ----- Ionisation source
iz_source = d["Sd+_iz"].squeeze()[:,2:-2]
iz_source = np.trapz(x = pos[2:-2], y = iz_source * J[2:-2] * d["Cs0"] * d["Nnorm"])

# ----- Recombination source
if "Sd+_rec" in d.keys():
    rec_source = d["Sd+_rec"].squeeze()[:,2:-2]
    rec_source = np.trapz(x = pos[2:-2], y = rec_source * J[2:-2] * d["Cs0"] * d["Nnorm"])
else:
    rec_source = np.zeros_like(input_source)


# ----- Totals
total_particles = d["Nd+"].squeeze()[:,2:-2] + d["Nd"].squeeze()[:,2:-2]
total_particles = np.trapz(x = pos[2:-2], y = total_particles * J[2:-2] * d["Nnorm"])
total_in = input_source + iz_source
total_out = sheath_ion_flux + abs(rec_source)
total_balance = total_in - total_out

neutral_in = rec_source
neutral_out = iz_source

fig, axes = plt.subplots(1,2, figsize=(10,4), dpi = 150)
t = range(len(iz_source))
ax = axes[0]
ax.plot(t, input_source, label = "Input source")
ax.plot(t, iz_source, label = "Ionisation source")
ax.plot(t, rec_source, label = "Recombination source")
ax.plot(t, sheath_ion_flux, ls = ":", c = "grey", label = "Ion sheath flux")
ax.set_ylabel("Particle flux [s-1]")

ax = axes[1]
ax.plot(t, total_particles, label = "Total domain particle count")
ax.set_ylabel("Particle count")

for ax in axes:
    ax.grid(which = "both")
    ax.set_xlabel("Timestep")
    ax.legend()

fig.show()

print(">>> System mass balance")
print("- Total in ---------------")
print(f"- Input source = {input_source[-1]:.3E} [s-1]")
print(f"- Ionisation source = {iz_source[-1]:.3E} [s-1]")
print(f"- Total = {total_in[-1]:.3E} [s-1]")
print("\n- Total out ---------------")
print(f"- Sheath ion flux = {sheath_ion_flux[-1]:.3E} [s-1]")
print(f"- Sheath neutral flux = {sheath_neutral_flux[-1]:.3E} [s-1]")
print(f"- Recombination source = {rec_source[-1]:.3E} [s-1]")
print(f"- Total = {total_out[-1]:.3E} [s-1]")
print(f"\n- Difference:")
print(f"---> {total_balance[-1]:.3E} [s-1] ({total_balance[-1]/total_in[-1]:.3%})")


