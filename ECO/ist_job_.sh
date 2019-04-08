#!/bin/bash

#SBATCH --job-name=ECO_
#SBATCH --output=ECO_.out
#SBATCH -p knm
##SBATCH -p p
#SBATCH -N 1
##SBATCH --cpus-per-task 20
#SBATCH c 200

srun python ./ECO_A_.py $@ >> ECO_A_.out
#srun ./ECO_A.sh

