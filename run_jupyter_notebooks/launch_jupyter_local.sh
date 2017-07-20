#!/bin/bash

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-local/bin/activate

# start jupyter
jupyter notebook --notebook-dir ~

# deactivate virtualenv
deactivate
