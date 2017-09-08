"""A deep MNIST classifier using convolutional layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tempfile
import math
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# ============================================================================ #
#                             DEFINE SOME VARIABLES                            #
# ============================================================================ #

tf.flags.DEFINE_integer('batch_size', 50, 
                        'Batch size per compute device.')
tf.flags.DEFINE_integer('train_steps', 20000, 
                        'Number of batches to run.')
tf.flags.DEFINE_integer('display_every', 10, 
                        '''Number of local steps after which progress is printed
                           out.''')
tf.flags.DEFINE_string('data_dir', './MNIST_data', 
                       'Path to dataset.')
tf.flags.DEFINE_string('data_format', 'NCHW', 
                       '''Data layout to use: NHWC (TF native) or NCHW (cuDNN  
                          native).''')
tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'Learning rate for training.')

FLAGS = tf.app.flags.FLAGS
train_dir = tempfile.mkdtemp()


# ============================================================================ #
#                            DEFINE SOME CONSTANTS                             #
# ============================================================================ #

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels and have 1 channel.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CH = 1

# Number of units in hidden layers.
HIDDEN1_UNITS = 32
HIDDEN2_UNITS = 64
HIDDEN3_UNITS = 1024


# ============================================================================ #
#                              BUILD MNIST GRAPHS                              #
# ============================================================================ #

def mnist_inference(images):
  """Build the MNIST model up to where it may be used for inference.
  Args:
      images: Images placeholder - [Batch_size, IMAGE_PIXELS].
  Returns:
      logits: Output tensor with the computed logits.
  """
  batch_size = tf.shape(images)[0]
  # Reshape to use within a convolutional neural net.
  with tf.name_scope('reshape'):
    if FLAGS.data_format == 'NCHW':
      reshaped_images = tf.reshape(images, 
                                   [batch_size, NUM_CH, IMAGE_SIZE, IMAGE_SIZE])
    else:
      reshaped_images = tf.reshape(images, 
                                   [batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CH])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
      weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CH, HIDDEN1_UNITS], 
                                                name='weights'))
      biases = tf.Variable(tf.zeros([HIDDEN1_UNITS]), name='biases')
      h_conv1 = tf.nn.relu(tf.nn.conv2d(reshaped_images, weights, 
                                        strides=[1, 1, 1, 1], padding='SAME', 
                                        data_format=FLAGS.data_format) + biases)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    ksize=[1, 2, 2, 1]
    strides=[1, 2, 2, 1]
    if FLAGS.data_format == 'NCHW':
      ksize=[ksize[0], ksize[3], ksize[1], ksize[2]]
      strides=[strides[0], strides[3], strides[1], strides[2]]
      h_pool1 = tf.nn.max_pool(h_conv1, ksize=ksize, strides=strides, 
                               padding='SAME', data_format=FLAGS.data_format)
    else:
      h_pool1 = tf.nn.max_pool(h_conv1, ksize=ksize, strides=strides, 
                               padding='SAME', data_format=FLAGS.data_format)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
      weights = tf.Variable(
          tf.truncated_normal([5, 5, HIDDEN1_UNITS, HIDDEN2_UNITS], 
                              name='weights'))
      biases = tf.Variable(tf.zeros([HIDDEN2_UNITS]), name='biases')
      h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, weights, 
                                        strides=[1, 1, 1, 1], padding='SAME', 
                                        data_format=FLAGS.data_format) + biases)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    ksize=[1, 2, 2, 1]
    strides=[1, 2, 2, 1]
    if FLAGS.data_format == 'NCHW':
      ksize=[ksize[0], ksize[3], ksize[1], ksize[2]]
      strides=[strides[0], strides[3], strides[1], strides[2]]
      h_pool2 = tf.nn.max_pool(h_conv2, ksize=ksize, strides=strides, 
                               padding='SAME', data_format=FLAGS.data_format)
    else:
      h_pool2 = tf.nn.max_pool(h_conv2, ksize=ksize, strides=strides, 
                               padding='SAME', data_format=FLAGS.data_format)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    weights = tf.Variable(
        tf.truncated_normal([(IMAGE_SIZE//4)*(IMAGE_SIZE//4)*HIDDEN2_UNITS, 
                             HIDDEN3_UNITS], name='weights'))
    biases = tf.Variable(tf.zeros([HIDDEN3_UNITS]), name='biases')
    h_pool2_flat = tf.reshape(h_pool2, [-1, (IMAGE_SIZE//4) * (IMAGE_SIZE//4) *
                                        HIDDEN2_UNITS])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob_placeholder = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_placeholder)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    weights = tf.Variable(tf.truncated_normal([HIDDEN3_UNITS, NUM_CLASSES], 
                                              name='weights'))
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    logits = tf.matmul(h_fc1_drop, weights) + biases

  return logits, keep_prob_placeholder


def mnist_accuracy(logits, labels):
  """Add to the Graph the Op that calculates the accuracy.
  Args:
      logits: Logits tensor, float - [Batch_size, NUM_CLASSES].
      labels: Labels placeholder - [Batch_size, NUM_CLASSES].
  Returns:
      accuracy: The Op for calculating accuracy.
  """
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

  return accuracy


def mnist_training(logits, labels, learning_rate):
  """Build the training graph.
  Args:
      logits: Logits tensor, float - [Batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [Batch_size], with values in the
        range [0, NUM_CLASSES).
      learning_rate: The learning rate to use for gradient descent.
  Returns:
      train_op: The Op for training.
      loss: The Op for calculating loss.
  """
  # Create an operation that calculates loss.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
      labels=labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)

  return train_op, loss


# ============================================================================ #
#                             TRAIN AND EVALUATION                             #
# ============================================================================ #

def train(train_data):

  # Build the complete graph for feeding inputs, training, and saving 
  # checkpoints.
  mnist_graph = tf.Graph()
  with mnist_graph.as_default():
    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])                                
    labels_placeholder = tf.placeholder(tf.int32, [None, NUM_CLASSES])
    tf.add_to_collection("images", images_placeholder)
    tf.add_to_collection("labels", labels_placeholder)

    # Build a Graph that computes predictions from the inference model.
    logits, keep_prob_placeholder = mnist_inference(images_placeholder)
    tf.add_to_collection("logits", logits)
    tf.add_to_collection("keep_prob", keep_prob_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op, loss = mnist_training(logits, labels_placeholder, 
                                    FLAGS.learning_rate)

    # Add to the Graph the Op that calculates the accuracy.
    accuracy_op = mnist_accuracy(logits, labels_placeholder)
    tf.add_to_collection("accuracy_op", accuracy_op)

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

  # Run training for MAX_STEPS and save checkpoint at the end.
  with tf.Session(graph=mnist_graph) as sess:
    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(FLAGS.train_steps):
        # Read a batch of images and labels.
        images_feed, labels_feed = train_data.next_batch(FLAGS.batch_size)

        # Run one step of the model. The return values are the activations
        # from the `train_op` (which is discarded), the `loss` and the 
        # `accuracy` Op. To inspect the values of your Ops or variables, you may
        # include them in the list passed to sess.run() and the value tensors 
        # will be returned in the tuple from the call.
        _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy_op], 
            feed_dict={images_placeholder: images_feed, 
                       labels_placeholder: labels_feed,
                       keep_prob_placeholder: 0.5})

        # Print out loss value.
        if step % FLAGS.display_every == 0:
          print('Step %d: loss = %.2f -- train accuracy = %g' % 
                (step, loss_value, accuracy_value))

    # Write a checkpoint.
    checkpoint_file = os.path.join(train_dir, 'checkpoint')
    saver.save(sess, checkpoint_file, global_step=step)


def eval(test_data, batch_size):

  # Run evaluation based on the saved checkpoint.
  with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(
        os.path.join(train_dir, "checkpoint-%d.meta" % (FLAGS.train_steps-1)))
    saver.restore(
        sess, os.path.join(train_dir, "checkpoint-%d" % (FLAGS.train_steps-1)))

    # Retrieve the Ops we 'remembered'.
    logits = tf.get_collection("logits")[0]
    images_placeholder = tf.get_collection("images")[0]
    labels_placeholder = tf.get_collection("labels")[0]
    keep_prob_placeholder = tf.get_collection("keep_prob")[0]
    accuracy_op = tf.get_collection("accuracy_op")[0]

    # Run evaluation.
    images_feed, labels_feed = test_data.next_batch(batch_size)

    print('test accuracy %g' % accuracy_op.eval(feed_dict={
        images_placeholder: images_feed, labels_placeholder: labels_feed, 
        keep_prob_placeholder: 1.0}))


# ============================================================================ #
#                                     MAIN                                     #
# ============================================================================ #

def main(_):
  argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  # Get input data: get the sets of images and labels for training, validation, 
  # and test on MNIST.
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  train(mnist.train)
  eval(mnist.test, mnist.test.num_examples)


if __name__ == '__main__':
  tf.app.run()
