#!/bin/bash

#SBATCH --job-name=distributed_tf
#SBATCH --time=00:12:00
#SBATCH --nodes=8
#SBATCH --constraint=gpu
#SBATCH --output=distributed_tf.%j.log

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.1.0-CrayGNU-2016.11-cuda-8.0-Python-3.5.2

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

# set TensorFlow script parameters
export TF_SCRIPT="$HOME/project_dir/project_script.py"

export TF_FLAGS="
--num_gpus=1 \
--batch_size=64 \
--num_batches=4 \
--data_format=NCHW \
--learning_rate=0.045 \
"

# set TensorFlow distributed parameters
export TF_NUM_PS=$1 # 1
export TF_NUM_WORKERS=$2 # $SLURM_JOB_NUM_NODES
# export TF_WORKER_PER_NODE=1
# export TF_PS_PER_NODE=1
# export TF_PS_IN_WORKER=true

# run distributed TensorFlow
DIST_TF_LAUNCHER_DIR=$SCRATCH/run_dist_tf_daint_directory
cd $DIST_TF_LAUNCHER_DIR
./run_dist_tf_daint.sh

# deactivate virtualenv
deactivate
