# Google Benchmarks

Here you can find the scripts that are used to generate the results shown in the [TensorFlow's benchmarks page](https://www.tensorflow.org/performance/benchmarks).
In order to create results that are as repeatable as possible, each test is run multiple (5 in our case) times and then the times are averaged together. For each test, 10 warmup steps are done and then the next 100 steps are averaged.

**Differences with the original source code:**
commented line 632 in `tf_cnn_benchmarks/tf_cnn_benchmarks.py` to discard the *force_gpu_compatible* option (rising some error on Piz Daint).


## Single node

### Piz Daint
Let USERNAME be your CSCS username.

1. `rsync -r tf_cnn_benchmarks util google-benchmarks_daint.sh run_benchmarks_daint.sh USERNAME@daint.cscs.ch:google-benchmarks`.
2. `ssh USERNAME@daint.cscs.ch`.
3. `./google-benchmarks/run_benchmarks_daint.sh <N>`.
This script runs `google-benchmarks_daint.sh` N times (if N is not passed, it defaults to 5), one after another, and prints the average number of trained images per second in each of the N executions on the screen.

### AWS
Let INSTANCE_IP be the public IP address of your EC2 instance.

If you want to use the ImageNet dataset, make sure you have copied it to your EC2 instance before running the bechmarks script, as described in `AWS.md` in `../environments_setup`.

1. `rsync -r tf_cnn_benchmarks util google-benchmarks_aws.sh run_benchmarks_aws.sh ubuntu@INSTANCE_IP:google-benchmarks`.
2. `ssh ubuntu@INSTANCE_IP`.
3. `./google-benchmarks/run_benchmarks_aws.sh <N>`.
This script runs `google-benchmarks_aws.sh` N times (if N is not passed, it defaults to 5), one after another, and prints the average number of trained images per second in each of the N executions on the screen.


## Multiple nodes (incl. one)

### Piz Daint
Let USERNAME be your CSCS username.

1. `scp google-benchmarks_dist_daint.sh run_daint_scripts.sh run_dist_benchmarks_daint.sh USERNAME@daint.cscs.ch:google-benchmarks`.
2. `ssh USERNAME@daint.cscs.ch mkdir google-benchmarks/outputs`.
3. `ssh USERNAME@daint.cscs.ch`.
4. `cd google-benchmarks`.
5. (Optional) `module load tmux; tmux`.
6. Modify `run_daint_scripts.sh` with your desired settings.
Make sure that the number of nodes you request in `google-benchmarks_dist_daint.sh` (SBATCH flag) is sufficient. In this file, you can also specify if you want Parameter Servers in the same nodes as the Workers (recommended) and the number of Workers and/or Parameter Servers per node (each GPU can be used by only one Worker/PS at a time).
7. `./run_daint_scripts.sh`.

### AWS
1. Update `../aws_private_ips.txt` and `../aws_public_ips.txt` with your instances' IPs.
2. `cd ../environments_setup`.
3. `./remote_setup_aws.sh` (takes ~5 minutes).
4. (Optional) `./check_remote_setup_aws.sh`.
5. (If you want to use data in S3) `./remote_aws_configure.sh`.
6. (If you want to use data in S3) `./remote_copy-dataset_aws.sh` (takes 30/60 minutes).
7. `cd ../google-benchmarks`.
8. `./remote_copy-code_aws.sh`.
9. Modify `remote_run_aws_scripts.sh` with your desired settings.
10. (Optional) `tmux`.
11. `./remote_run_aws_scripts.sh`.

You can also run AWS scripts remotely from Daint, provided you copy the private key you use to access EC2 instances into Daint:

1. `scp ~/.ssh/AWS_PRIVATEKEY USERNAME@daint.cscs.ch:.ssh`.
2. `ssh USERNAME@daint.cscs.ch mkdir aws aws/outputs aws/code`.
3. `scp ../aws_p* daint.cscs.ch:aws`.
4. `scp remote_google-benchmarks_dist_aws.sh remote_run_aws_scripts.sh remote_run_dist_benchmarks_aws.sh USERNAME@daint.cscs.ch:aws/code`.
5. `ssh USERNAME@daint.cscs.ch`.
6. `cd aws/code`.
7. (Optional) `module load tmux; tmux`.
8. `./remote_run_aws_scripts.sh`.
