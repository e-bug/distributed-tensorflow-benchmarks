#!/bin/bash

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-local/bin/activate

# save dependencies
pip3 freeze > all_requirements.txt

# remove tensorflow dependency (to avoid conflicts with tensorflow-gpu)
grep -v '^tensorflow' all_requirements.txt > requirements.txt
rm all_requirements.txt

# deactivate virtualenv
deactivate
