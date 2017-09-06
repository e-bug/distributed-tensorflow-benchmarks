#!/bin/bash

# Arguments:
#   $1: job_name: either "ps" or "worker"
#   $2: task_index: index of `job_name` instance (starting from 0)
#   $3: TF_PS_HOSTS: Parameter Servers' hostnames
#   $4: TF_WORKER_HOSTS: Workers' hostnames
#   $5: variable_update: parameter_server/distributed_replicated
#   $6: real_data: true/false

# set TensorFlow script parameters
TF_DIST_FLAGS=" --ps_hosts=$3 --worker_hosts=$4"

TF_SCRIPT_DIR=$HOME/Dropbox/cscs/tensorflow-benchmarks/google-benchmarks/tf_cnn_benchmarks
TF_SCRIPT=$TF_SCRIPT_DIR/tf_cnn_benchmarks.py

data_flags="
--eval=False \
--save_summaries_steps=2 \
--summary_verbosity=1 \
--train_dir=train_dir \
--num_gpus=1 \
--batch_size=8 \
--num_warmup_batches=1 \
--num_batches=4 \
--display_every=1 \
--data_format=NHWC \
--use_nccl=False \
--variable_update=$5 \
--local_parameter_device=cpu \
--device=cpu \
--model=inception3 \
--optimizer=rmsprop \
--learning_rate=0.045 \
--num_epochs_per_decay=2 \
--gradient_clip=2.0 \
--data_name=imagenet \
--data_dir=
"
nodata_flags="
--eval=False \
--save_summaries_steps=2 \
--summary_verbosity=1 \
--train_dir=train_dir \
--num_gpus=1 \
--batch_size=8 \
--num_warmup_batches=1 \
--num_batches=4 \
--display_every=1 \
--data_format=NHWC \
--use_nccl=False \
--variable_update=$5 \
--local_parameter_device=cpu \
--device=cpu \
--model=inception3 \
--optimizer=rmsprop \
--learning_rate=0.045 \
--num_epochs_per_decay=2 \
--gradient_clip=2.0 \
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
source $WORKON_HOME/tf-local/bin/activate

# train inception
python ${TF_SCRIPT} --job_name=$1 --task_index=$2 ${TF_DIST_FLAGS} ${TF_FLAGS}

# deactivate virtualenv
deactivate
