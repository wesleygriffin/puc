#!/bin/bash
#SBATCH --job-name="vadd"
#SBATCH --output="out/sbatch.out"
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --export=ALL
#SBATCH -t 7-0

FORCE_TCP="--mca pml ob1 --mca btl self,tcp"
mpiexec -v $FORCE_TCP --output-filename vadd.out ./vadd -N 1000000

