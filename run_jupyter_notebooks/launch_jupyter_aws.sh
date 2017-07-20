#!/bin/bash

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-aws/bin/activate

# start jupyter
jupyter notebook --no-browser --ip 0.0.0.0 --notebook-dir ~

# deactivate virtualenv
deactivate
