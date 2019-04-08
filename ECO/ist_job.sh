#!/bin/bash

#SBATCH --job-name=ECO
#SBATCH --output=ECO.out
#SBATCH -p p
#SBATCH -N 1
##SBATCH --cpus-per-task 20
#SBATCH c 20

srun python ./ECO_A.py $@ >> ECO_A.out
#srun ./ECO_A.sh

