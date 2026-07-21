#!/bin/bash
#SBATCH --account=ctb-simine
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G

module load gcc openmpi flexiblas fftw eigen voro++
export PATH=$HOME/software/lammps-mtp-cpu/bin:$PATH

lmp -in npt_si_300k.in
