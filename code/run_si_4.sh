#!/bin/bash
#SBATCH --account=ctb-simine
#SBATCH --time=13:00:00
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1

module load gcc openmpi flexiblas fftw eigen voro++
export PATH=$HOME/software/lammps-mtp-cpu/bin:$PATH

cd hold
srun lmp -in hold_4.in