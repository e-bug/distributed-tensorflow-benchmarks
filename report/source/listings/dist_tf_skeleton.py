import sys
import tensorflow as tf

# Specify the cluster's architecture
cluster = tf.train.ClusterSpec({'ps': ['192.168.1.1:1111'],
                                'worker': ['192.168.1.2:1111',
                                           '192.168.1.3:1111']
                               })

# Parse command-line to specify machine
job_type = sys.argv[1]  # job type: "worker" or "ps"
task_idx = sys.argv[2]  # index job in the worker or ps list
                        # as defined in the ClusterSpec

# Create TensorFlow Server. This is how the machines communicate.
server = tf.train.Server(cluster, job_name=job_type, task_index=task_idx)

# Parameter server is updated by remote clients.
# Will not proceed beyond this if statement.
if job_type == 'ps':
  server.join()
else:
  # Workers only
  with tf.device(tf.train.replica_device_setter(
                      worker_device='/job:worker/task:'+task_idx,
                      cluster=cluster)):
    # Build your model here as if you only were using a single machine

  with tf.Session(server.target):
    # Train your model here
