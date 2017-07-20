#!/bin/bash

# install virtualenv
sudo apt install virtualenv

# create virtualenv
export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
cd $WORKON_HOME
virtualenv -p /usr/bin/python3.5m tf-local

# load virtualenv
source $WORKON_HOME/tf-local/bin/activate

# install TensorFlow 1.0.0
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp35-cp35m-linux_x86_64.whl

# install jupyter
pip3 install --upgrade jupyter

# deactivate virtualenv
deactivate
