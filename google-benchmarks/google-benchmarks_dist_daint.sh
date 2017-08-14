#!/bin/bash

#SBATCH --job-name=google_benchmark
#SBATCH --time=00:12:00
#SBATCH --nodes=128
#SBATCH --constraint=gpu
#SBATCH --output=dist_benchmark_daint.%j.log

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: variable_update: parameter_server/distributed_replicated
#   $4: real_data: true/false

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.1.0-CrayGNU-2016.11-cuda-8.0-Python-3.5.2

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

# set TensorFlow script parameters
export TF_SCRIPT="$HOME/google-benchmarks/tf_cnn_benchmarks/tf_cnn_benchmarks.py"

data_flags="
--num_gpus=1 \
--batch_size=64 \
--data_format=NCHW \
--use_nccl=True \
--variable_update=$3 \
--local_parameter_device=cpu \
--optimizer=sgd \
--model=inception3 \
--data_name=imagenet \
--data_dir=/scratch/snx3000/maximem/deeplearnpackages/ImageNet/TF/
"
nodata_flags="
--num_gpus=1 \
--batch_size=64 \
--data_format=NCHW \
--use_nccl=True \
--variable_update=$3 \
--local_parameter_device=cpu \
--optimizer=sgd \
--model=inception3 \
--data_name=imagenet
"
if [ "$4" = "true" ]; then
  export TF_FLAGS=$data_flags
elif [ "$4" = "false" ]; then
  export TF_FLAGS=$nodata_flags
else
  echo "error in real_data argument"
  exit 1
fi

# set TensorFlow distributed parameters
export TF_NUM_PS=$1
export TF_NUM_WORKERS=$2 # $SLURM_JOB_NUM_NODES
# export TF_WORKER_PER_NODE=1
# export TF_PS_PER_NODE=1
# export TF_PS_IN_WORKER=true

# run distributed TensorFlow
DIST_TF_LAUNCHER_DIR=$HOME/google-benchmarks
cd $DIST_TF_LAUNCHER_DIR
./run_dist_tf.sh

# deactivate virtualenv
deactivate
