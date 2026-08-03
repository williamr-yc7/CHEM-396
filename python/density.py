import os
import numpy as np

# ---------------------------------------------------------------
# Density from a LAMMPS slurm/log file.
# Reads the Volume column of the thermo output, averages it, and
# converts to g/cm^3 for a system of N silicon atoms.
# ---------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(script_dir, "density")
os.makedirs(outdir, exist_ok=True)

M_SI = 28.0855          # molar mass of silicon, g/mol
N_A  = 6.02214e23       # Avogadro, 1/mol

# 1. Locate the slurm file
path = input("slurm output path: ")
path = os.path.abspath(os.path.join(script_dir, path))
print(f"Looking for file at: {path}")

natoms = int(input("Number of atoms (e.g. 216): "))
skip = float(input("Skip the first how many ps as equilibration (e.g. 100): "))

# 2. Read the thermo table.
#    thermo_style custom step temp press vol pe gives 5 numeric columns.
#    Any line that is not 5 numbers (headers, timing tables, warnings) is skipped.
steps, vols = [], []
with open(path) as f:
    for line in f:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue          # header line such as "Step Temp Press Volume PotEng"
        steps.append(row[0])
        vols.append(row[3])   # Volume is the 4th column

steps = np.array(steps)
vols  = np.array(vols)

if len(vols) == 0:
    raise SystemExit("No thermo data found. Check the file and the thermo_style columns.")

# 3. Drop the equilibration period. Steps are fs, so skip*1000 steps = skip ps.
mask = steps >= skip * 1000
vols = vols[mask]
if len(vols) == 0:
    raise SystemExit("Skipped everything. Lower the equilibration window.")

# 4. Average and convert. The 1e-24 turns Angstrom^3 into cm^3.
mean_v  = vols.mean()
sem     = vols.std(ddof=1) / np.sqrt(len(vols))    # standard error of the mean
rho     = natoms * M_SI / (N_A * mean_v * 1e-24)
rho_err = rho * sem / mean_v                       # same relative error

print(f"Mean volume: {mean_v:.1f} +/- {sem:.1f} A^3")
print(f"Density: {rho:.4f} +/- {rho_err:.4f} g/cm^3")