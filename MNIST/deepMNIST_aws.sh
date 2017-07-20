#!/bin/bash

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-aws/bin/activate

# run script
cd ~/MNIST
mkdir -p outputs
python deepMNIST.py > outputs/deepMNIST_aws.log

# deactivate virtualenv
deactivate

