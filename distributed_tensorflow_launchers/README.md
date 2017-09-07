# Distributed TensorFlow Launchers

Here you can find scripts that help you launch distributed TensorFlow applications on your local workstation, on Piz Daint or on Amazon EC2 instances.

For each system, execute the *setup* script to start your application in a distributed mode.
For Piz Daint, you specify everything in its setup script, while some parameters are present in the *run_dist_tf* files for the other cases.

The distributed MNIST tutorial and the benchmarking scripts make use of these launchers.

## Local
- `setup_dist_tf_local.sh`: launches one process per PS/Worker on different terminal windows. Each process calls `run_dist_tf_local.sh` with its job type and index. Arguments: number of Parameter Servers and number of Workers.
- `run_dist_tf_local.sh`: starts the Python script with the flags herein specified for each process.

## Piz Daint
Python code needs to have: `job_name`, `task_index`, `ps_hosts` and `worker_hosts` TensorFlow flags.
They will be specified by `run_dist_tf_daint.sh` once nodes for the job are allocated.

- `setup_dist_tf_daint.sh`: submits a Slurm job by calling `run_dist_tf_daint.sh` for the Python script with the flags herein specified. Arguments: number of Parameter Servers and number of Workers.
- `run_dist_tf_daint.sh`: distribute the tasks to the allocated nodes by creating a script for each node (in order to run both a PS and a Worker on a single node if needed).

## AWS
- `remote_setup_dist_tf_aws.sh`: launches PS/Worker processes on the specified instances (via their IPs) by starting a screen session for each of them in which `run_dist_tf_aws.sh` is executed.
- `run_dist_tf_aws.sh`: starts the Python script with the flags herein specified for each process.
