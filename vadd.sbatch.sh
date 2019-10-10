#!/bin/bash
#SBATCH --job-name="vadd"
#SBATCH --output="out/sbatch.out"
#SBATCH --partition=gpu
#SBATCH --gpu-bind=closest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH -t 7-0

FORCE_TCP="--mca pml ob1 --mca btl self,tcp"
mpiexec -v $FORCE_TCP --output-filename out ./vadd -N 1000000

