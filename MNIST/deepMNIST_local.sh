#!/bin/bash

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-local/bin/activate

# run script
mkdir -p outputs
python deepMNIST.py > outputs/deepMNIST_local.log

# deactivate virtualenv
deactivate
