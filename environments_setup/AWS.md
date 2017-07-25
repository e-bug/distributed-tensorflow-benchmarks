# AWS

Amazon Web Services (AWS) provides on-demand cloud computing platforms on a paid subscription basis. The AWS technology allows subscribers to have at their disposal a full-fledged virtual cluster of computers, available all the time, through the internet. It is implemented at server farms throughout the world.

The most popular AWS services include Amazon Elastic Compute Cloud (EC2) and Amazon Simple Storage Service (S3).

## Creating an EC2 instance with Spot Requests

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

## Deleting instances and Spot Requests

Deleting a Spot Request and all its instances:

1. Go to Amazon Web Services.
2. In *All services*, click on *EC2* under *Compute*.
3. Click on *Spot Requests* under *INSTANCES* in the left column.
4. Select your Spot Request in the list.
5. Click the *Actions* button first, and then *Cancel Spot request*.

Deleting a single instance:

1. Go to Amazon Web Services.
2. In *All services*, click on *EC2* under *Compute*.
3. Click on *Instances* under *INSTANCES* in the left column.
4. Select your instance in the list.
5. Click the *Actions* button first, and then *Instance State* and *Terminate*.

## Create an S3 bucket

1. Connect to Amazon Web Services.
2. In *All services*, click on *S3* under *Storage*.
3. Click on the *Create bucket* button.
4. Choose a name for the bucket (Note: no uppercase letters); and its Region.
5. Click on *Next* and proceed with the creation.

Make sure that the Region of your bucket is the same as the Region where you launch your EC2 instances. In fact, *there is no Data Transfer charge for data transferred between Amazon EC2 and Amazon S3 within the same Region*.

## [aws-cli](https://github.com/aws/aws-cli)

This package provides a unified command line interface to Amazon Web Services.

aws-cli is installed during the initial setup.

Before using aws-cli, you need to tell it about your AWS credentials. 
The quickest way to get started is to run the `aws configure` command (after you have activatated your virtual environment):
```
(tf-daint) USERNAME@daint102:~> aws configure
AWS Access Key ID [None]: foo
AWS Secret Access Key [None]: bar
Default region name [None]: us-west-2
Default output format [None]: json
```
Use your bucket region name for `Default region name`; you can find the code of each region at http://docs.aws.amazon.com/general/latest/gr/rande.html.
You can leave `Default output format` empty.

## Transferring data from/to S3

Use the following command to copy an object from Amazon S3 to your machine:
`aws s3 cp s3://my_bucket/my_folder/my_file.ext my_copied_file.ext`.

Use the following command to copy an object from your machine into Amazon S3:
`aws s3 cp my_copied_file.ext s3://my_bucket/my_folder/my_file.ext`.

Use the following command to download an entire Amazon S3 bucket to a local directory on your machine:
`aws s3 sync s3://remote_S3_bucket local_directory`.

### Transferring ImageNet from Piz Daint to S3
After you have configured `aws` on Daint as described above, you can copy the ImageNet dataset to S3 as follows:

1. Run `ssh USERNAME@daint.cscs.ch`.
2. (Optional) Run `module load tmux; tmux`.
3. Run `source Envs/tf-daint/bin/activate`.
4. Run `aws s3 cp --recursive /scratch/snx3000/maximem/deeplearnpackages/ImageNet/TF/ s3://tfcscs/imagenet`.
This creates the `imagenet` directory and copies all the files in it.

### Transferring ImageNet from S3 to EC2
After you have configured `aws` on your EC2 instance as described above, you can copy the ImageNet dataset to your instance as follows:

1. Run `source Envs/tf-aws/bin/activate`.
2. Run `aws s3 cp --recursive s3://tfcscs/imagenet ~/imagenet`.
This creates the `imagenet` directory and copies all the files in it.