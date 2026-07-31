import os
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Mean square displacement for a LAMMPS trajectory.
# Flat MSD = atoms vibrating in place (solid).
# Linear MSD = atoms diffusing (liquid).
#
# Two modes:
#   1 = single trajectory. Used for the melt runs, where the curve is flat
#       and then turns sharply upward at the moment of melting.
#   2 = several trajectories overlaid. Used for the holds, which are
#       isothermal and linear throughout, so the comparison across
#       potentials is made on the slope rather than on the shape.
# ---------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(script_dir, "msd")
os.makedirs(outdir, exist_ok=True)

def msd_curve(u):
    """MSD against time for a whole trajectory, in A^2."""
    # The dump stores wrapped coordinates, so an atom crossing the box edge
    # appears to jump the full box length. We undo that by comparing each frame
    # to the previous one and subtracting whole box lengths from any jump larger
    # than half a box. Valid because atoms move far less than L/2 per frame.
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

    # MSD relative to the starting configuration, averaged over all atoms
    disp = unwrapped - unwrapped[0]
    return np.mean(np.sum(disp**2, axis=2), axis=1)

def diffusion(time, msd, frac=0.5):
    """Self-diffusion coefficient in m^2/s from the Einstein relation.

    Fits a straight line to the final `frac` of the curve and divides the
    slope by 6 (that is 2d for d = 3 dimensions). Only meaningful for an
    equilibrated liquid: on a heating ramp the system is still changing
    temperature, so the slope is not a diffusion coefficient.
    """
    cut = int(len(time) * (1 - frac))
    slope = np.polyfit(time[cut:], msd[cut:], 1)[0]   # A^2 per ps
    return slope / 6 * 1e-8                            # convert to m^2/s

def load(path):
    """Open a LAMMPS dump as an MDAnalysis Universe."""
    full = os.path.abspath(os.path.join(script_dir, path))
    print(f"Looking for file at: {full}")
    return mda.Universe(full, format="LAMMPSDUMP", topology_format="LAMMPSDUMP",
                        lammps_coordinate_convention="scaled")

# ---------------------------------------------------------------
mode = input("Mode: 1 = single file (melt), 2 = multiple files (holds): ").strip()

# Time between dumped frames, in ps (dump every 100 steps at 1 fs = 0.1 ps)
dt = float(input("Time between frames in ps (e.g. 0.1): "))

datasets = []      # list of (label, time array, msd array)

if mode == "1":
    u = load(input("dump.lammpstrj path: "))
    msd = msd_curve(u)
    datasets.append(("", np.arange(len(msd)) * dt, msd))
    colours = ["blue"]

else:
    nfiles = int(input("How many trajectories? "))
    for i in range(nfiles):
        path  = input(f"  [{i+1}] dump.lammpstrj path: ")
        label = input(f"  [{i+1}] label for the legend (e.g. 22.almtp): ")
        u = load(path)
        msd = msd_curve(u)
        datasets.append((label, np.arange(len(msd)) * dt, msd))
    colours = ["red", "blue", "green", "orange", "purple", "brown", "black"]

# ---------------------------------------------------------------
filename = input("Name the output without extension: ")

# Summary: endpoint and diffusion coefficient for each trajectory
with open(os.path.join(outdir, filename + "_summary.txt"), "w") as f:
    for label, time, msd in datasets:
        D = diffusion(time, msd)
        line = (f"{label if label else 'trajectory'}: final MSD = {msd[-1]:.2f} A^2 "
                f"after {time[-1]:.1f} ps, D = {D:.3e} m^2/s")
        print(line)
        f.write(line + "\n")

# Full curves. One column per trajectory, sharing the time axis of the first.
with open(os.path.join(outdir, filename + ".txt"), "w") as f:
    f.write("# Time(ps)" + "".join(f"    MSD_{lbl if lbl else 'A'}(A^2)" for lbl, _, _ in datasets) + "\n")
    nrows = min(len(t) for _, t, _ in datasets)
    for r in range(nrows):
        f.write(f"{datasets[0][1][r]:12.4f}" + "".join(f" {m[r]:12.4f}" for _, _, m in datasets) + "\n")
print(f"Data saved to {filename}.txt")

# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
for (label, time, msd), col in zip(datasets, colours):
    plt.plot(time, msd, color=col, linewidth=2, label=label if label else None)
plt.xlabel("Time (ps)")
plt.ylabel(r"MSD ($\AA^2$)")
plt.title("Mean Square Displacement")
if mode != "1":
    plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(outdir, filename + ".png"), dpi=300)
plt.show()