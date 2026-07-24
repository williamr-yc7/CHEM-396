#!/bin/bash
#SBATCH --account=ctb-simine
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

module load gcc openmpi flexiblas fftw eigen voro++ cuda
export PATH=$HOME/software/lammps-mtp-gpu/bin:$PATH

lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in test.in
