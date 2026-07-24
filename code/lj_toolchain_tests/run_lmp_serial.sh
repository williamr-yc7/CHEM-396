#!/bin/bash

#SBATCH --account=ctb-simine
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0-00:30

module load StdEnv/2020 intel/2020.1.217 openmpi/4.0.3 lammps-omp/20210929

lmp -in test.in