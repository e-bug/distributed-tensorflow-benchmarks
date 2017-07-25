#!/bin/bash

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-aws/bin/activate

# train inception
cd ~/google-benchmarks
python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--num_gpus=1 \
--batch_size=32 \
--data_format=NCHW \
--use_nccl=True \
--variable_update=parameter_server \
--local_parameter_device=cpu \
--optimizer=sgd \
--model=inception3 \
--data_name=imagenet \
#--data_dir=/home/ubuntu/imagenet/

# deactivate virtualenv
deactivate

