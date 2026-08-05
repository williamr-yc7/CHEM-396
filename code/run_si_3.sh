#!/bin/bash
#SBATCH --account=ctb-simine
#SBATCH --time=09:00:00
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1

module load gcc openmpi flexiblas fftw eigen voro++
export PATH=$HOME/software/lammps-mtp-cpu/bin:$PATH

cd quench
srun lmp -in quench_3.in