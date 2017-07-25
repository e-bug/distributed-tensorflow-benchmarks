#!/bin/bash

#SBATCH --job-name=test_tf
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --mem=60GB
#SBATCH --output=test_tf.%j.log

# load modules
module load daint-gpu
module load TensorFlow/1.0.0-CrayGNU-2016.11-cuda-8.0-Python-3.5.2

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

# run test
python -c "import tensorflow as tf;hello = tf.constant('Hello, TensorFlow');sess = tf.Session();print(sess.run(hello))"

# deactivate virtualenv
deactivate

