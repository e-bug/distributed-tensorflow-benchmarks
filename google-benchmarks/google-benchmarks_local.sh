#!/bin/bash

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-local/bin/activate

# train inception
SCRIPT_DIR=$HOME/Dropbox/cscs/tensorflow-benchmarks/google-benchmarks
cd $SCRIPT_DIR
python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--num_gpus=1 \
--batch_size=8 \
--num_warmup_batches=2 \
--num_batches=10 \
--data_format=NHWC \
--variable_update=parameter_server \
--local_parameter_device=cpu \
--device=cpu \
--optimizer=sgd \
--model=inception3 \
--data_name=imagenet \
#--data_dir=/home/ubuntu/imagenet/

# deactivate virtualenv
deactivate

