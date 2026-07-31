import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Tetrahedral order parameter q for a LAMMPS trajectory.
# q = 1 for perfect tetrahedral geometry, 0 for a random arrangement.
# Chau & Hardwick (1998), normalization of Errington & Debenedetti (2001).
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
outdir = os.path.join(script_dir, "top")
os.makedirs(outdir, exist_ok=True)

def tetra(u, start, stop):
    """Tetrahedral order parameter of every atom in every frame from start to stop."""
    out = []
    for ts in u.trajectory[start:stop]:
        p = u.atoms.positions
        box = u.dimensions[:3]
        # Full distance matrix with periodic boundary conditions applied
        d = distance_array(p, p, box=u.dimensions)
        np.fill_diagonal(d, np.inf)          # an atom is not its own neighbour
        for i in range(len(p)):
            # The 4 nearest neighbours, regardless of how far away they are. No cutoff here: q is about the shape of the neighbourhood, not its size.
            nn = np.argsort(d[i])[:4]
            # Vectors from atom i to each neighbour, minimum image so that neighbours across the box edge point the right way
            v = p[nn] - p[i]
            v -= box * np.round(v / box)
            v /= np.linalg.norm(v, axis=1)[:, None]   # unit vectors, lengths discarded
            q = 1.0
            # The 6 distinct neighbour pairs (j<k out of 4)
            for j in range(3):
                for k in range(j+1, 4):
                    cos = np.clip(np.dot(v[j], v[k]), -1, 1)
                    q -= (3/8) * (cos + 1/3)**2       # zero for the ideal 109.47 deg
            out.append(q)
    return np.array(out)

def load(path):
    """Open a LAMMPS dump as an MDAnalysis Universe."""
    full = os.path.abspath(os.path.join(script_dir, path))
    print(f"Looking for file at: {full}")
    return mda.Universe(full, format="LAMMPSDUMP", topology_format="LAMMPSDUMP",
                        lammps_coordinate_convention="scaled")

# ---------------------------------------------------------------
mode = input("Mode: 1 = single file (melt, two windows), 2 = multiple files (holds): ").strip()

datasets = []      # list of (label, q array), whatever the mode

if mode == "1":
    u = load(input("dump.lammpstrj path: "))
    n = len(u.trajectory)
    datasets.append(("First 2 ps (crystal)", tetra(u, 0, 20)))
    datasets.append(("Last 3 ps (liquid)",   tetra(u, n - 30, n)))
    colours = ["red", "blue"]

else:
    # Each file contributes one curve, computed over the same trailing window
    # so the comparison across potentials is like for like.
    nfiles = int(input("How many trajectories? "))
    nframes = int(input("How many frames from the end of each run? "))
    for i in range(nfiles):
        path  = input(f"  [{i+1}] dump.lammpstrj path: ")
        label = input(f"  [{i+1}] label for the legend (e.g. 22.almtp): ")
        u = load(path)
        n = len(u.trajectory)
        datasets.append((label, tetra(u, n - nframes, n)))
    colours = ["red", "blue", "green", "orange", "purple", "brown", "black"]

# ---------------------------------------------------------------
# Save the summary. Crystal should sit just below 1, liquid well below.
filename = input("Name the output without extension: ")
with open(os.path.join(outdir, filename + ".txt"), "w") as f:
    for label, q in datasets:
        line = f"{label}: mean q = {q.mean():.4f}, std = {q.std():.4f}"
        print(line)
        f.write(line + "\n")
print(f"Data saved to {filename}.txt")

# ---------------------------------------------------------------
# Distribution of q. Crystal narrow near 1, liquid broad and shifted down.
plt.figure(figsize=(6, 4))
bins = np.linspace(-0.5, 1.0, 60)
for (label, q), c in zip(datasets, colours):
    if mode == "1":
        plt.hist(q, bins=bins, density=True, alpha=0.6, color=c, label=label)
    else:
        # Filled histograms overlap badly with more than two datasets,
        # so the multi-file mode draws outlines instead.
        plt.hist(q, bins=bins, density=True, histtype="step", linewidth=2,
                 color=c, label=label)
plt.xlabel("Tetrahedral order parameter q")
plt.ylabel("Probability density")
plt.title("Tetrahedral Order Parameter Distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(outdir, filename + ".png"), dpi=300)
plt.show()