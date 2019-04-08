#!/bin/bash

#SBATCH --job-name=B_ECO
#SBATCH --output=B_ECO.out
#SBATCH -p p
#SBATCH -N 1
#SBATCH c 20
#SBATCH --mem 200GB
srun python ./ECO_B.py $@ >> ECO_B.out

