import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import self_capped_distance
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Coordination number analysis for a LAMMPS trajectory.
# Counts neighbours within a bond cutoff.
#
# Two modes:
#   1 = single trajectory, split into an early and a late window.
#       Used for the melt runs, where the same trajectory contains both
#       the crystal and the liquid.
#   2 = several trajectories, one window each, overlaid.
#       Used for the holds, which are isothermal throughout, so there is
#       no within-run contrast and the comparison is made across potentials.
# ---------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(script_dir, "coord_num")
os.makedirs(outdir, exist_ok=True)

def coord(u, start, stop, cutoff):
    """Coordination number of every atom in every frame from start to stop."""
    out = []
    for ts in u.trajectory[start:stop]:
        # All unique pairs closer than the cutoff. box= applies periodic boundary conditions, so neighbours across the box edge still count.
        pairs = self_capped_distance(u.atoms.positions, cutoff, box=u.dimensions, return_distances=False)
        # Each pair appears once as (i,j). Flattening and counting how often each index shows up gives the neighbour count per atom.
        out.append(np.bincount(pairs.ravel(), minlength=len(u.atoms)))
    return np.concatenate(out)

def load(path):
    """Open a LAMMPS dump as an MDAnalysis Universe."""
    full = os.path.abspath(os.path.join(script_dir, path))
    print(f"Looking for file at: {full}")
    return mda.Universe(full, format="LAMMPSDUMP", topology_format="LAMMPSDUMP",
                        lammps_coordinate_convention="scaled")

# ---------------------------------------------------------------
mode = input("Mode: 1 = single file (melt, two windows), 2 = multiple files (holds): ").strip()

# Bond cutoff, taken from the first minimum of g(r). 2.85 A is the convention
# used by Zongo et al. for a-Si; the liquid minimum sits nearer 3.15 A.
cutoff = float(input("Bond cutoff in Angstroms: "))

datasets = []      # list of (label, coordination array), whatever the mode

if mode == "1":
    u = load(input("dump.lammpstrj path: "))
    n = len(u.trajectory)
    datasets.append(("First 2 ps (crystal)", coord(u, 0, 20, cutoff)))
    datasets.append(("Last 3 ps (liquid)",   coord(u, n - 30, n, cutoff)))
    colours = ["red", "blue"]

else:
    # Each file contributes one histogram, computed over the same trailing
    # window so the comparison across potentials is like for like.
    nfiles = int(input("How many trajectories? "))
    nframes = int(input("How many frames from the end of each run? "))
    for i in range(nfiles):
        path  = input(f"  [{i+1}] dump.lammpstrj path: ")
        label = input(f"  [{i+1}] label for the legend (e.g. 22.almtp): ")
        u = load(path)
        n = len(u.trajectory)
        datasets.append((label, coord(u, n - nframes, n, cutoff)))
    colours = ["red", "blue", "green", "orange", "purple", "brown", "black"]

# ---------------------------------------------------------------
# Mean coordination and the breakdown by coordination number. Crystal should
# give exactly 4.000 and 100% 4-fold. Liquid should rise toward 6 with a broad spread.
filename = input("Name the output without extension: ")
with open(os.path.join(outdir, filename + ".txt"), "w") as f:
    f.write(f"# Bond cutoff: {cutoff} A\n")
    for label, c in datasets:
        header = f"\n{label}: mean = {c.mean():.3f}"
        print(header)
        f.write(header + "\n")
        for k in range(2, 12):
            frac = 100 * np.mean(c == k)
            if frac > 0.1:              # skip empty bins
                line = f"   {k}-fold: {frac:.1f}%"
                print(line)
                f.write(line + "\n")
print(f"\nData saved to {filename}.txt")

# ---------------------------------------------------------------
# Histogram. Bin edges offset by 0.5 so each integer gets its own bar.
bins = np.arange(1.5, 12.5)
plt.figure(figsize=(6, 4))
for (label, c), col in zip(datasets, colours):
    if mode == "1":
        plt.hist(c, bins=bins, density=True, alpha=0.6, color=col, label=label)
    else:
        # Filled bars overlap badly with more than two datasets,
        # so the multi-file mode draws outlines instead.
        plt.hist(c, bins=bins, density=True, histtype="step", linewidth=2,
                 color=col, label=label)
plt.xlabel("Coordination number")
plt.ylabel("Fraction of atoms")
plt.title("Coordination Number Distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(outdir, filename + ".png"), dpi=300)
plt.show()