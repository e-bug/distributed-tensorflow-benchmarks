#!/bin/bash

# load virtualenv
source $HOME/Envs/tf-local/bin/activate

# run MNIST
python deepMNIST_gpu.py \
--batch_size=50 \
--train_steps=100 \
--data_format=NHWC \
--display_every=10 \
--data_dir=./MNIST_data

# deactivate virtualenv
deactivate
