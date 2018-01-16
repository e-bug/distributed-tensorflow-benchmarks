#!/bin/bash
#SBATCH --job-name=mlcomm
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --exclusive
#SBATCH --output=evaluation-%j

echo "Number of nodes: " 1

# . /opt/modules/default/init/bash

# setup our environment
module load daint-gpu
module load TensorFlow/1.3.0-CrayGNU-17.08-cuda-8.0-python3
module load craype-ml-plugin-py3/1.0.1

srun -u --exclusive --cpu_bind=none /scratch/snx3000/${USER}/cray-tensorflow/evaluate_checkpoints.sh \
    /scratch/snx3000/${USER}/cray-tf-traindir/rank0 \
    /scratch/snx3000/${USER}/cray-tensorflow/tmp/


