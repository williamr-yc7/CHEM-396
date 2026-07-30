import os
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import matplotlib.pyplot as plt

# 1. Load your LAMMPS trajectory file
# Note: 'format="LAMMPS"' handles standard .lammpstrj or .dump files

# Get the directory where your python script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

drc = input("dump.lammpstrj path: ")
drc = os.path.abspath(os.path.join(script_dir, drc))
print(f"Looking for file at: {drc}")
# Change your universe initialization line to include the coordinate convention:
u = mda.Universe(
    drc, 
    format="LAMMPSDUMP", 
    topology_format="LAMMPSDUMP", 
    lammps_coordinate_convention="scaled"
)

# 2. Select the atom groups you want to analyze
# "all" selects every atom. You can also use "type 1", "type 2", etc.
atoms_A = u.select_atoms("all")
atoms_B = u.select_atoms("all")

# 3. Configure the RDF Analysis
# 'nbins' is the number of histogram bins
# 'range' sets the minimum and maximum distance in Angstroms (0 to cutoff)
max_dist = float(input("Input max distance 'cutoff' in Angstroms: "))
rdf_analyzer = rdf.InterRDF(atoms_A, atoms_B, nbins=100, range=(0.0, max_dist))

# 4. Run the calculation across all frames in the dump file
print("Calculating RDF... This may take a moment for large files.")
rdf_analyzer.run()

# 5. Extract results
radii = rdf_analyzer.results.bins
g_r = rdf_analyzer.results.rdf

# 6. Save the calculated RDF data to a text file
filename = input("Name the output.txt without extension: ")
with open("python/radial_dist_fct/"+filename+".txt", "w") as f:
    f.write("# Radius(A)    g(r)\n")
    for r, g in zip(radii, g_r):
        f.write(f"{r:12.4f} {g:12.4f}\n")
print("Data successfully saved to 'python_rdf_output.txt'")

# 7. Plot the results
plt.figure(figsize=(6, 4))
plt.plot(radii, g_r, label="All-All RDF", color="blue", linewidth=2)
plt.xlabel("Distance r ($\AA$)")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function")
plt.grid(True)
plt.legend()
plt.savefig("python/radial_dist_fct/"+filename+".png", dpi=300)
plt.show()
