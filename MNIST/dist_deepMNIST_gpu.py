"""A distributed deep MNIST classifier using convolutional layers.
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
tf.flags.DEFINE_integer('num_gpus', 1, 
                        '''Number of gpus for each machine. If you don't use 
                           GPU, please set it to 0.''')
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

# Distributed training flags.
tf.flags.DEFINE_string('job_name', '',
                       'One of "ps" or "worker".')
tf.flags.DEFINE_string('ps_hosts', '', 
                       'Comma-separated list of hostname:port hosts.')
tf.flags.DEFINE_string('worker_hosts', '',
                       'Comma-separated list of hostname:port hosts.')
tf.flags.DEFINE_integer('task_index', None, 
                        'Index of task within the job.')
tf.flags.DEFINE_boolean('sync_replicas', False,
                        '''Use the sync_replicas (synchronized replicas) mode,
                           wherein the parameter updates from workers are 
                           aggregated before applied to avoid stale gradients.
                        ''')
tf.flags.DEFINE_integer('replicas_to_aggregate', None,
                        '''Number of replicas to aggregate before parameter 
                           update is applied (For sync_replicas mode only; 
                           default: num_workers).''')

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
      conv = tf.nn.relu(tf.nn.conv2d(reshaped_images, weights, 
                                     strides=[1, 1, 1, 1], padding='SAME', 
                                     data_format=FLAGS.data_format))
      conv_shape = conv.get_shape().as_list()

      h_conv1 = tf.reshape(tf.nn.bias_add(conv, biases, 
                                          data_format=FLAGS.data_format),
                           [batch_size, conv_shape[1], 
                            conv_shape[2], conv_shape[3]])

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
      conv = tf.nn.relu(tf.nn.conv2d(h_pool1, weights, 
                                     strides=[1, 1, 1, 1], padding='SAME', 
                                     data_format=FLAGS.data_format))
      conv_shape = conv.get_shape().as_list()
      h_conv2 = tf.reshape(tf.nn.bias_add(conv, biases,
                                          data_format=FLAGS.data_format),
                           [batch_size, conv_shape[1], 
                            conv_shape[2], conv_shape[3]])

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


def mnist_training(logits, labels, learning_rate, num_workers):
  """Build the training graph.
  Args:
      logits: Logits tensor, float - [Batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [Batch_size], with values in the
        range [0, NUM_CLASSES).
      learning_rate: The learning rate to use for gradient descent.
  Returns:
      train_op: The Op for training.
      loss: The Op for calculating loss.
      opt: The Op for optimizing.
  """
  # Create an operation that calculates loss.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
      labels=labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

  # Create the gradient descent optimizer with the given learning rate.
  opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

  if FLAGS.sync_replicas:
    if FLAGS.replicas_to_aggregate is None:
      replicas_to_aggregate = num_workers
    else:
      replicas_to_aggregate = FLAGS.replicas_to_aggregate

    opt = tf.train.SyncReplicasOptimizer(opt, 
              replicas_to_aggregate=replicas_to_aggregate,
              total_num_replicas=num_workers,
              name="mnist_sync_replicas")

  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = opt.minimize(loss, global_step=global_step)

  return train_op, loss, global_step, opt


# ============================================================================ #
#                       DISTRIBUTED TRAIN AND EVALUATION                       #
# ============================================================================ #

def train_and_eval(train_data, server, num_workers, is_chief, 
                   test_data, test_batch_size):

  # Build the complete graph for feeding inputs, training, and saving 
  # checkpoints.

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
  train_op, loss, global_step, opt = mnist_training(logits, 
                                                    labels_placeholder,
                                                    FLAGS.learning_rate,
                                                    num_workers)

  # Add to the Graph the Op that calculates the accuracy.
  accuracy_op = mnist_accuracy(logits, labels_placeholder)
  tf.add_to_collection("accuracy_op", accuracy_op)

  # Add the variable initializer Op.
  if FLAGS.sync_replicas:
    if is_chief:
      local_init_op = opt.chief_init_op
    else: 
      local_init_op = opt.local_step_init_op
    ready_for_local_init_op = opt.ready_for_local_init_op
    # Initial token and chief queue runners required by the sync_replicas mode
    chief_queue_runner = opt.get_chief_queue_runner()
    sync_init_op = opt.get_init_tokens_op()

  init_op = tf.global_variables_initializer()

  # Create a saver for writing training checkpoints.
  saver = tf.train.Saver()

  # Create Supervisor that takes care of common needs of TensorFlow training 
  # programs.
  if FLAGS.sync_replicas:
    sv = tf.train.Supervisor(is_chief=is_chief, 
                             logdir=train_dir, 
                             init_op=init_op, 
                             local_init_op=local_init_op,
                             ready_for_local_init_op=ready_for_local_init_op,
                             recovery_wait_secs=1, 
                             global_step=global_step)
  else:
    sv = tf.train.Supervisor(is_chief=is_chief, 
                             logdir=train_dir,
                             init_op=init_op, 
                             recovery_wait_secs=1, 
                             global_step=global_step)

  # The chief worker (task_index==0) session will prepare the session,
  # while the remaining workers will wait for the preparation to complete.
  sess_config = tf.ConfigProto(allow_soft_placement=True, 
                               log_device_placement=False)
  if is_chief:
    print('Worker %d: Initializing session...' % FLAGS.task_index)
  else:
    print('Worker %d: Waiting for session to be initialized...' %
          FLAGS.task_index)
  sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
  print('Worker %d: Session initialization complete.' % FLAGS.task_index)

  if FLAGS.sync_replicas and is_chief:
    # Chief worker will start the chief queue runner and call the init op.
    sess.run(sync_init_op)
    sv.start_queue_runners(sess, [chief_queue_runner])

  # Start the training loop.
  local_step = 0
  while not sv.should_stop():
    # Read a batch of images and labels.
    images_feed, labels_feed = train_data.next_batch(FLAGS.batch_size)

    # Run one step of the model.
    _, step, loss_value, accuracy_value = sess.run(
        [train_op, global_step, loss, accuracy_op], 
        feed_dict={images_placeholder: images_feed, 
                   labels_placeholder: labels_feed,
                   keep_prob_placeholder: 0.5})
    local_step += 1
      
    if step >= FLAGS.train_steps:
      break

    # Print out loss value.
    if step % FLAGS.display_every == 0:
      print('Step %d: loss = %.2f -- train accuracy = %g' % 
            (local_step, loss_value, accuracy_value))


  if is_chief:
    # Write a checkpoint.
    checkpoint_file = os.path.join(train_dir, 'checkpoint')
    saver.save(sess, checkpoint_file, global_step=step)


  if is_chief:
    # Run evaluation.
    images_feed, labels_feed = test_data.next_batch(test_batch_size)

    print('Test accuracy = %g' % accuracy_op.eval(session=sess,
          feed_dict={images_placeholder: images_feed, 
                     labels_placeholder: labels_feed, 
                     keep_prob_placeholder: 1.0}))


# ============================================================================ #
#                                   CLUSTER                                    #
# ============================================================================ #

def define_cluster():
  """Construct the cluster.
  Returns:
      cluster: Specified cluster's architecture.
      server: TensorFlow Server, for a given ps.
      num_workers: Number of worker hosts.
  """

  # Specify the cluster's architecture.
  ps_spec = FLAGS.ps_hosts.split(',')
  worker_spec = FLAGS.worker_hosts.split(',')

  # Get the number of workers.
  num_workers = len(worker_spec)
  
  cluster = tf.train.ClusterSpec({
      'ps': ps_spec,
      'worker': worker_spec})

  # Create TensorFlow Server. This is how the machines communicate.
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, 
                           task_index=FLAGS.task_index) 

  return cluster, server, num_workers


def start_server(server):
  if FLAGS.job_name == 'ps':
    print('Running PS %d...' % FLAGS.task_index)
    server.join() # blocking


def run_worker(mnist, cluster, server, num_workers):

  print('Starting Worker %d.' % FLAGS.task_index)

  is_chief = (FLAGS.task_index == 0)
  if FLAGS.num_gpus > 0:
    # Allocate task_index->#gpu for each worker in the corresponding machine.
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # By default all CPUs available to the process are aggregated under cpu:0.
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                                cluster=cluster)):
    train_and_eval(mnist.train, 
                   server, num_workers, is_chief, 
                   mnist.test, mnist.test.num_examples)


# ============================================================================ #
#                                     MAIN                                     #
# ============================================================================ #

def print_info():
  """Print basic information."""
  print('Batch size:  %s global' % (FLAGS.batch_size * FLAGS.num_gpus))
  print('             %s per device' % FLAGS.batch_size)
  print('Train steps  %d' % FLAGS.train_steps)
  print('PSs:         %s' % FLAGS.ps_hosts)
  print('WORKERs:     %s' % FLAGS.worker_hosts)
  print('Data format: %s' % FLAGS.data_format)
  print('Sync:        %s' % FLAGS.sync_replicas)
  print('==========')


def main(_):
  argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Get input data: get the sets of images and labels for training, validation, 
  # and test on MNIST.
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Assert existence of distributed flags.
  if FLAGS.job_name is None or FLAGS.job_name == '':
    raise ValueError('Must specify an explicit `job_name`')
  if FLAGS.task_index is None or FLAGS.task_index == '':
    raise ValueError('Must specify an explicit `task_index`')

  print_info()

  # Specify the architecture of the cluster.
  cluster, server, num_workers = define_cluster()
  # Start server.
  start_server(server)
  # Start worker and run application.
  run_worker(mnist, cluster, server, num_workers)


if __name__ == '__main__':
  tf.app.run()
