# Initial TensorFlow setup

Here, you can find how the initial setup required to run TensorFlow in the different systems under study.
The chosen version of TensorFlow is 1.0.0 (same as in Piz Daint).


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

#### Launch an Instance
1. Go to your Amazon Web Services Sign-in page ([here](https://701314272525.signin.aws.amazon.com/console) in our case) and log in using your AWS username and password.
2. In *All services*, click on *EC2* under *Compute*.
3. Click on *Spot Requests* under *INSTANCES* in the left column.
4. Press the *Request Spot Instances* button at the top of the page to open your request form.
5. In this Step 1, select *Canonical, Ubuntu, 16.04 LTS, amd64 xenial image* for **AMI**.
6. Select your instance type. For example, *p2.xlarge* if you want a compute node with a single NVIDIA K80 GPU. 
All the instance types are described [here](https://aws.amazon.com/ec2/instance-types/). In particular, look at the *Accelerated Computing* tab.
7. Click *Next* at the bottom to proceed to Step 2.
8. Here, ask for more than 8 GB (e.g., 16 GB) in **EBS volumes**.
9. Enable **EBS-optimized**.
10. Choose a public-private key pair in **Key pair name** among the ones you have shared.
11. Click on *Choose from a list* in **IAM fleet role** and select your role. In our case, *spot_fleet_role*.
12. Click on *Review* at the end of the page.
13. Click on *Launch* at the end of the opened Step 3 page.
14. A *Success* pop-up should appear; click *OK* to proceed.
15. Click on the *Instances* tab at mid page, and then click on the link of your *Instance Id*.
16. Finally, in the *Description* tab, you can find the IP address of this instance under the key *IPv4 Public IP*.

#### Setup TensorFlow
Still assume your CSCS username is USERNAME.
Also, let INSTANCE_IP be the IP address of the instance you launched.

1. Run `scp USERNAME@daint.cscs.ch:/apps/daint/UES/6.0.UP02/sandbox-ds/soures/cudnn-8.0-linux-x64-v5.1.tgz .` to retrieve cuDNN from Piz Daint.
2. Run `scp requirements.txt cudnn-8.0-linux-x64-v5.1.tgz setup_virtenv_aws.sh test_tf_aws.sh ubuntu@INSTANCE_IP:` to copy all the required files to the AWS instance.
3. Run `ssh ubuntu@INSTANCE_IP` to log into the AWS instance.
4. In the AWS instance, just run `./setup_virtenv_aws.sh` and press `Enter` or `Y` whenever requested (it takes 5 minutes).
5. (Optional) Run `./test_tf_aws.sh` to check that TensorFlow is working. You should see "Hello, TensorFlow".


## Distributed
