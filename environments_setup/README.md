# Initial setup

Here, you can find the initial setup required to run TensorFlow in the different systems under study.
The chosen version of TensorFlow is 1.1.0.


## One node

Start setting up your local machine and then move to the remote ones.

### Local
1. Run `./setup_virtenv_local.sh` to create a virtual environment called *tf-local*.
2. Run `./get_requirements.sh` to produce a file (*requirements.txt*) containing all the requirements for the other systems.

3. (Optional) Run `./test_tf_local.sh` to check that TensorFlow is working. You should see "Hello, TensorFlow".

### Piz Daint
In the following, it is assumed that your CSCS username is USERNAME.

1. Run `scp requirements.txt setup_virtenv_daint.sh test_tf_daint.sh USERNAME@daint.cscs.ch:` to copy all the required files to Piz Daint.
2. Run `ssh USERNAME@daint.cscs.ch` to log into Piz Daint.
3. In Daint, just run `./setup_virtenv_daint.sh`.
5. (Optional) Run `sbatch test_tf_daint.sh` to check that TensorFlow is working. You should see "Hello, TensorFlow" in a file called *test_tf.XXXXXX.log*.

### AWS
Refer to `AWS.md` to see how to create an EC2 instance.

Still assume your CSCS username is USERNAME, and let INSTANCE_IP be the IP address of the instance you launched.

1. Run `scp USERNAME@daint.cscs.ch:/apps/daint/UES/6.0.UP02/sandbox-ds/soures/cudnn-8.0-linux-x64-v5.1.tgz .` to retrieve cuDNN from Piz Daint.
2. Run `scp requirements.txt cudnn-8.0-linux-x64-v5.1.tgz setup_virtenv_aws.sh test_tf_aws.sh ubuntu@INSTANCE_IP:` to copy all the required files to the AWS instance.
3. Run `ssh ubuntu@INSTANCE_IP` to log into the AWS instance.
4. In the AWS instance, just run `./setup_virtenv_aws.sh` and press `Enter` or `Y` whenever requested (it takes 5 minutes).
5. (Optional) Run `./test_tf_aws.sh` to check that TensorFlow is working. You should see "Hello, TensorFlow".


## Distributed
