#!/bin/bash

# load modules
module load daint-gpu
module load TensorFlow/1.0.0-CrayGNU-2016.11-cuda-8.0-Python-3.5.2

# create virtualenv
export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
cd $WORKON_HOME
virtualenv tf-daint

# load virtualenv
source $WORKON_HOME/tf-daint/bin/activate

# install dependencies
pip install -r ../requirements.txt

# install aws-cli
pip install awscli

# deactivate virtualenv
deactivate
