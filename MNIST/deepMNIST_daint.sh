#!/bin/bash

#SBATCH --job-name=deepMNIST
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --mem=60GB
#SBATCH --output=deepMNIST_daint.%j.log

# load modules
module load daint-gpu
module load TensorFlow/1.0.0-CrayGNU-2016.11-cuda-8.0-Python-3.5.2

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

# run script
cd ~/MNIST
srun -C gpu python deepMNIST.py

# deactivate virtualenv
deactivate
