#!/bin/bash

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-local/bin/activate

# run test
python -c "import tensorflow as tf;hello = tf.constant('Hello, TensorFlow');sess = tf.Session();print(sess.run(hello))"

# deactivate virtualenv
deactivate

