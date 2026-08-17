import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import self_capped_distance
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Check if the atoms making the large excursions the defective ones?
# Keeps coordination and displacement as (frame, atom) arrays so the two can be matched by index, which the separate scripts do not allow
# ---------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

def load(path):
    # Open a LAMMPS dump as an MDAnalysis Universe.
    full = os.path.abspath(os.path.join(script_dir, path))
    print(f"Looking for file at: {full}")
    return mda.Universe(full, format="LAMMPSDUMP", topology_format="LAMMPSDUMP", lammps_coordinate_convention="scaled")

u = load(input("dump.lammpstrj path: "))
cutoff  = float(input("Bond cutoff in Angstroms: ")) # 2.85 for the quenched solid
nframes = int(input("How many frames from the end? "))
n = len(u.trajectory)

coord_list, pos_list, box_list = [], [], []
for ts in u.trajectory[n - nframes:]:
    pairs = self_capped_distance(u.atoms.positions, cutoff, box=u.dimensions, return_distances=False)
    coord_list.append(np.bincount(pairs.ravel(), minlength=len(u.atoms))) # neighbours per atom, this frame
    pos_list.append(u.atoms.positions.copy()) # copy, MDAnalysis reuses the array
    box_list.append(ts.dimensions[:3].copy())

coord = np.array(coord_list) # (nframes, natoms)
pos   = np.array(pos_list) # (nframes, natoms, 3)
box   = np.array(box_list) # (nframes, 3)

d = np.diff(pos, axis=0) # step between consecutive frames
d -= box[1:, None, :] * np.round(d / box[1:, None, :]) # round is +-1 only where an atom wrapped
unwrapped = np.concatenate([pos[:1], pos[:1] + np.cumsum(d, axis=0)], axis=0) # rebuild continuous paths

unwrapped -= unwrapped.mean(axis=1, keepdims=True) # drop the drift of the box as a whole
disp = np.linalg.norm(unwrapped - unwrapped.mean(axis=0, keepdims=True), axis=2) # distance from each atom's own site
print(f"\nFrames x atoms: {disp.shape}, mean amplitude {disp.mean():.3f} A")

threshold = 0.5
is_defect = coord != 4 # non-fourfold, dangling or floating
in_tail   = disp > threshold # moments an atom sits beyond the voxel size

baseline = 100 * is_defect.mean()
tail     = 100 * is_defect[in_tail].mean()

print(f"\nSample points in the tail: {in_tail.sum()} of {in_tail.size}")
print(f"Defective overall:        {baseline:.1f}%")
print(f"Defective in the tail:    {tail:.1f}%")
print(f"Enrichment:               {tail / baseline:.1f}x")

for k in (3, 5): # threefold is short a neighbour, so need not behave like fivefold
    overall = 100 * np.mean(coord == k)
    intail  = 100 * np.mean(coord[in_tail] == k)
    print(f"  {k}-fold: {overall:.1f}% overall, {intail:.1f}% in the tail")

idx, counts = np.unique(np.where(in_tail)[1], return_counts=True)
print(np.c_[idx, counts])

for i in [21, 177, 91, 139]:
    print(i, coord[:, i].min(), coord[:, i].max(), np.mean(coord[:, i] == 3))