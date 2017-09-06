# Initial setup

Here, you can find the initial setup required to run TensorFlow in the different systems under study.
The chosen version of TensorFlow is 1.1.0.

Start setting up your local machine and then set up the remote ones.

## Local
1. Run `./setup_virtenv_local.sh` to create a virtual environment called *tf-local*.
2. Run `./get_requirements.sh` to produce a file (*requirements.txt*) containing all the requirements for the other systems.

3. (Optional) Run `./test_tf_local.sh` to check that TensorFlow is working. You should see "Hello, TensorFlow".

## Piz Daint
Let USERNAME be your CSCS username is USERNAME.

1. Run `scp requirements.txt setup_virtenv_daint.sh test_tf_daint.sh USERNAME@daint.cscs.ch:` to copy all the required files to Piz Daint.
2. Run `ssh USERNAME@daint.cscs.ch` to log into Piz Daint.
3. In Daint, just run `./setup_virtenv_daint.sh`.
5. (Optional) Run `sbatch test_tf_daint.sh` to check that TensorFlow is working. You should see "Hello, TensorFlow" in a file called *test_tf.SLUMR_JOBID.log*.

## AWS
Refer to `AWS.md` to see how to create an EC2 instance.

Still assume your CSCS username is USERNAME.

1. Create a file `../aws_public_ips.txt` containing the public IP addresses of all your EC2 instances, one per line.
2. Run `scp USERNAME@daint.cscs.ch:/apps/daint/UES/6.0.UP02/sandbox-ds/soures/cudnn-8.0-linux-x64-v5.1.tgz .` to retrieve cuDNN from Piz Daint.
3. Run `remote_setup_aws.sh`. It takes 5 minutes per machine to finish.
4. (Optional) Run `./check_remote_setup_aws.sh` to check if all your instances have been correctly set up (after waiting around 5 minutes from the previous step).
5. (Optional) If you plan to use S3, run `./remote_aws_configure.sh` in order to configure aws-cli.
6. (Optional) If you want to copy the ImageNet dataset from S3 to all your instances, run `./remote_copy-dataset_aws.sh`.