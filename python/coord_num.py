import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import self_capped_distance
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Coordination number analysis for a LAMMPS melt trajectory.
# Counts neighbours within a bond cutoff, compares crystal vs liquid.
# ---------------------------------------------------------------

# 1. Locate the dump file relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
drc = input("dump.lammpstrj path: ")
drc = os.path.abspath(os.path.join(script_dir, drc))
print(f"Looking for file at: {drc}")

# 2. Load the trajectory. 'scaled' because LAMMPS dumps xs ys zs by default
u = mda.Universe(drc, format="LAMMPSDUMP", topology_format="LAMMPSDUMP", lammps_coordinate_convention="scaled")

# 3. Bond cutoff, taken from the first minimum of g(r). 2.85 A is the convention used by Zongo et al. for silicon.
cutoff = float(input("Bond cutoff in Angstroms: "))

def coord(start, stop):
    """Coordination number of every atom in every frame from start to stop."""
    out = []
    for ts in u.trajectory[start:stop]:
        # All unique pairs closer than the cutoff. box= applies periodic boundary conditions, so neighbours across the box edge still count.
        pairs = self_capped_distance(u.atoms.positions, cutoff, box=u.dimensions, return_distances=False)
        # Each pair appears once as (i,j). Flattening and counting how often each index shows up gives the neighbour count per atom.
        out.append(np.bincount(pairs.ravel(), minlength=len(u.atoms)))
    return np.concatenate(out)

# 4. Same two windows used for the RDF: start of the ramp vs end
n = len(u.trajectory)
cold = coord(0, 20)        # first 2 ps, still crystalline
hot  = coord(n - 30, n)    # last 3 ps, molten

# 5. Mean coordination and the breakdown by coordination number. Crystal should give exactly 4.000 and ~100% 4-fold. Liquid should rise toward 6 with a broad spread.
filename = input("Name the output.txt without extension: ")
with open("python/coord_num/"+filename+".txt", "w") as f:
    f.write(f"# Bond cutoff: {cutoff} A\n")
    for name, c in (("First 2 ps (crystal)", cold), ("Last 3 ps (liquid)", hot)):
        header = f"\n{name}: mean = {c.mean():.3f}"
        print(header)
        f.write(header + "\n")
        for k in range(2, 10):
            frac = 100 * np.mean(c == k)
            if frac > 0.1:              # skip empty bins
                line = f"   {k}-fold: {frac:.1f}%"
                print(line)
                f.write(line + "\n")
print(f"\nData saved to {filename}.txt")

# 6. Histogram. Bin edges offset by 0.5 so each integer gets its own bar.
bins = np.arange(1.5, 10.5)
plt.figure(figsize=(6, 4))
plt.hist(cold, bins=bins, density=True, alpha=0.6, color="red",  label="First 2 ps (crystal)")
plt.hist(hot,  bins=bins, density=True, alpha=0.6, color="blue", label="Last 3 ps (liquid)")
plt.xlabel("Coordination number")
plt.ylabel("Fraction of atoms")
plt.title("Coordination Number Distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("python/coord_num/"+filename+".png", dpi=300)
plt.show()