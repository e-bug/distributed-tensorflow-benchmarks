#!/bin/bash

# Arguments:
#   $1: job_name: either "ps" or "worker"
#   $2: task_index: index of `job_name` instance (starting from 0)
#   $3: TF_PS_HOSTS: Parameter Servers' hostnames
#   $4: TF_WORKER_HOSTS: Workers' hostnames
#   $5: variable_update: parameter_server/distributed_replicated
#   $6: real_data: true/false
#   $7: NUM_GPU: number of GPUs per worker

# set TensorFlow script parameters
TF_DIST_FLAGS=" --ps_hosts=$3 --worker_hosts=$4"

TF_SCRIPT="$HOME/tf_cnn_benchmarks/tf_cnn_benchmarks.py"

data_flags="
--num_gpus=$7 \
--batch_size=64 \
--data_format=NCHW \
--use_nccl=True \
--variable_update=$5 \
--local_parameter_device=cpu \
--optimizer=sgd \
--model=inception3 \
--data_name=imagenet \
--data_dir=/home/ubuntu/imagenet/
"
nodata_flags="
--num_gpus=$7 \
--batch_size=64 \
--data_format=NCHW \
--use_nccl=True \
--variable_update=$5 \
--local_parameter_device=cpu \
--optimizer=sgd \
--model=inception3 \
--data_name=imagenet
"
if [ "$6" = "true" ]; then
  TF_FLAGS=$data_flags
elif [ "$6" = "false" ]; then
  TF_FLAGS=$nodata_flags
else
  echo "error in real_data argument"
  exit 1
fi

# load virtualenv
export WORKON_HOME=$HOME/Envs
source $WORKON_HOME/tf-aws/bin/activate

# train inception
n_gpus=`nvidia-smi -L | wc -l`
cvd=`seq -s ',' $counter 0 $(($n_gpus-1))`
if [ $1 = "worker" ]; then
  CUDA_VISIBLE_DEVICES=${cvd}
else
  CUDA_VISIBLE_DEVICES=
fi
python ${TF_SCRIPT} --job_name=$1 --task_index=$2 ${TF_DIST_FLAGS} ${TF_FLAGS}

# deactivate virtualenv
deactivate
