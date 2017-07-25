# Google Benchmarks

Here you can find the scripts that are used to generate the results shown in the [TensorFlow's benchmarks page](https://www.tensorflow.org/performance/benchmarks).
In order to create results that are as repeatable as possible, each test is run multiple (5 in our case) times and then the times were averaged together. For each test, 10 warmup steps are done and then the next 100 steps are averaged.

Differences with original source code: 
commented line 632 in `tf_cnn_benchmarks/tf_cnn_benchmarks.py` to discard the `force_gpu_compatible` option in TensorFlow 1.0.0.

## Piz Daint
Let USERNAME be your CSCS username.

1. `rsync -r tf_cnn_benchmarks util google-benchmarks_daint.sh run_parallel_benchmarks_daint.sh run_serial_benchmarks_daint.sh USERNAME@daint.cscs.ch:google-benchmarks`.
2. `ssh USERNAME@daint.cscs.ch`.
3. `./google-benchmarks/run_parallel_benchmarks_aws.sh <N>` OR `./google-benchmarks/run_serial_benchmarks_aws.sh <N>`.
These scripts run `google-benchmarks_aws.sh` N times (if N is not passed, it defaults to 5) and print the average number of trained images per second in each of the N executions on the screen.
The former script submits N jobs at the same time (recommended when synthetic data is used), while the latter submit them one after another.

## AWS
Let INSTANCE_IP be the IP address of your EC2 instance.

Before running the bechmarks script, make sure you have copied the ImageNet dataset into your EC2 instance, as described in `AWS.md` in `../environments_setup`.

1. `rsync -r tf_cnn_benchmarks util google-benchmarks_aws.sh run_serial_benchmarks_aws.sh ubuntu@INSTANCE_IP:google-benchmarks`.
2. `ssh ubuntu@INSTANCE_IP`.
3. `./google-benchmarks/run_serial_benchmarks_aws.sh <N>`.
This script runs `google-benchmarks_aws.sh` N times (if N is not passed, it defaults to 5), one after another, and prints the average number of trained images per second in each of the N executions on the screen.