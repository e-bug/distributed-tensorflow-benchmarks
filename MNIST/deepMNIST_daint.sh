#!/bin/bash

#SBATCH --job-name=deepMNIST
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --output=deepMNIST_daint.%j.log

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.1.0-CrayGNU-2016.11-cuda-8.0-Python-3.5.2

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

# run script
cd $HOME/MNIST
srun python deepMNIST.py

# deactivate virtualenv
deactivate
