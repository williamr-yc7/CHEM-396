import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Radial distribution function g(r) for a LAMMPS trajectory.
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
outdir = os.path.join(script_dir, "radial_dist_fct")
os.makedirs(outdir, exist_ok=True)

def load(path):
    """Open a LAMMPS dump as an MDAnalysis Universe."""
    full = os.path.abspath(os.path.join(script_dir, path))
    print(f"Looking for file at: {full}")
    return mda.Universe(full, format="LAMMPSDUMP", topology_format="LAMMPSDUMP",
                        lammps_coordinate_convention="scaled")

def compute(u, start, stop, max_dist):
    """g(r) over the frames from start to stop.

    exclusion_block=(1,1) removes self-pairs. Without it every atom is paired
    with itself at r = 0, and dividing by the near-zero shell volume of the
    first bin produces a spurious spike thousands of times the true g(r).
    """
    atoms = u.select_atoms("all")
    analyzer = rdf.InterRDF(atoms, atoms, nbins=100, range=(0.0, max_dist),
                            exclusion_block=(1, 1))
    analyzer.run(start=start, stop=stop)
    return analyzer.results.bins.copy(), analyzer.results.rdf.copy()

# ---------------------------------------------------------------
mode = input("Mode: 1 = single file (melt, two windows), 2 = multiple files (holds): ").strip()

max_dist = float(input("Input max distance 'cutoff' in Angstroms: "))

datasets = []      # list of (label, radii, g(r))

if mode == "1":
    u = load(input("dump.lammpstrj path: "))
    # Under periodic boundary conditions g(r) is only defined out to half the
    # box length, since beyond that the minimum image no longer samples a full
    # spherical shell.
    half_box = u.dimensions[:3].min() / 2
    if max_dist > half_box:
        print(f"Warning: cutoff {max_dist} exceeds L/2 = {half_box:.2f} A, truncating")
        max_dist = half_box

    n = len(u.trajectory)
    print("Calculating RDF... This may take a moment for large files.")
    r, g = compute(u, 0, 20, max_dist)
    datasets.append(("First 2 ps (crystal)", r, g))
    r, g = compute(u, n - 30, n, max_dist)
    datasets.append(("Last 3 ps (liquid)", r, g))
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
        half_box = u.dimensions[:3].min() / 2
        if max_dist > half_box:
            print(f"Warning: cutoff {max_dist} exceeds L/2 = {half_box:.2f} A, truncating")
            max_dist = half_box
        n = len(u.trajectory)
        print(f"Calculating RDF for {label}...")
        r, g = compute(u, n - nframes, n, max_dist)
        datasets.append((label, r, g))
    colours = ["red", "blue", "green", "orange", "purple", "brown", "black"]

# ---------------------------------------------------------------
filename = input("Name the output without extension: ")

# One column of g(r) per dataset, sharing the radius axis of the first
with open(os.path.join(outdir, filename + ".txt"), "w") as f:
    f.write("# Radius(A)" + "".join(f"    g_{lbl}" for lbl, _, _ in datasets) + "\n")
    for row in range(len(datasets[0][1])):
        f.write(f"{datasets[0][1][row]:12.4f}" + "".join(f" {g[row]:12.4f}" for _, _, g in datasets) + "\n")

# Peak positions, the numbers that go straight into the report table
for label, r, g in datasets:
    # First peak: highest point of the curve
    i1 = np.argmax(g)
    # First minimum after it, then the second peak beyond that
    i_min = i1 + np.argmin(g[i1:])
    i2 = i_min + np.argmax(g[i_min:]) if i_min < len(g) - 1 else i_min
    print(f"{label}: r1 = {r[i1]:.2f} A (g = {g[i1]:.2f}), "
          f"first min = {r[i_min]:.2f} A (g = {g[i_min]:.2f}), "
          f"r2 = {r[i2]:.2f} A (g = {g[i2]:.2f})")
print(f"Data saved to {filename}.txt")

# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
for (label, r, g), col in zip(datasets, colours):
    plt.plot(r, g, label=label, color=col, linewidth=2)
plt.xlabel(r"Distance r ($\AA$)")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(outdir, filename + ".png"), dpi=300)
plt.show()