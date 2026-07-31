import os
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Mean square displacement for a LAMMPS melt trajectory.
# Flat MSD = atoms vibrating in place (solid).
# Linear MSD = atoms diffusing (liquid).
# ---------------------------------------------------------------

# 1. Locate the dump file relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
drc = input("dump.lammpstrj path: ")
drc = os.path.abspath(os.path.join(script_dir, drc))
print(f"Looking for file at: {drc}")

# 2. Load the trajectory
u = mda.Universe(drc, format="LAMMPSDUMP", topology_format="LAMMPSDUMP", lammps_coordinate_convention="scaled")

# 3. Time between dumped frames, in ps (dump every 100 steps at 1 fs = 0.1 ps)
dt = float(input("Time between frames in ps (e.g. 0.1): "))

# 4. Unwrap the trajectory.
#    The dump stores wrapped coordinates, so an atom crossing the box edge
#    appears to jump the full box length. We undo that by comparing each frame
#    to the previous one and subtracting whole box lengths from any jump larger
#    than half a box. Valid because atoms move far less than L/2 per frame.
pos = []
for ts in u.trajectory:
    pos.append(u.atoms.positions.copy())
pos = np.array(pos)                      # shape (frames, atoms, 3)
box = u.dimensions[:3]                   # cubic box, edge lengths

unwrapped = np.zeros_like(pos)
unwrapped[0] = pos[0]
for t in range(1, len(pos)):
    delta = pos[t] - pos[t-1]
    delta -= box * np.round(delta / box)  # minimum image
    unwrapped[t] = unwrapped[t-1] + delta

# 5. MSD relative to the starting configuration, averaged over all atoms
disp = unwrapped - unwrapped[0]
msd = np.mean(np.sum(disp**2, axis=2), axis=1)
time = np.arange(len(msd)) * dt

# 6. Save data and report the endpoint
filename = input("Name the output.txt without extension: ")
with open("python/msd/"+filename+".txt", "w") as f:
    f.write("# Time(ps)      MSD(A^2)\n")
    for t, m in zip(time, msd):
        f.write(f"{t:12.4f} {m:12.4f}\n")
print(f"Final MSD: {msd[-1]:.2f} A^2 after {time[-1]:.1f} ps")
print(f"Data saved to {filename}.txt")

# 7. Plot
plt.figure(figsize=(6, 4))
plt.plot(time, msd, color="blue", linewidth=2)
plt.xlabel("Time (ps)")
plt.ylabel(r"MSD ($\AA^2$)")
plt.title("Mean Square Displacement")
plt.grid(True, alpha=0.3)
plt.savefig("python/msd/"+filename+".png", dpi=300)
plt.show()