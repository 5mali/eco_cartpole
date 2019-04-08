#!/bin/bash

#SBATCH --job-name=B1_ECO
#SBATCH --output=B1_ECO.out
#SBATCH -p p
#SBATCH -N 1
#SBATCH -n 11
#SBATCH --mem 150GB
srun python ./ECO_B1.py $@ >> ECO_B1.data

