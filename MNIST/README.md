# Running the MNIST tutorial

You can run the MNIST tutorial in two ways:

- in an interactive way through the `DeepMNIST.ipynb` notebook;
- with a script that calls `deepMNIST.py`.

In the former case, just launch a Jupyter notebook (as described in [`run_jupyter_notebooks/`](../run_jupyter_notebooks)) and browse to the MNIST directory.
For the latter, just run the corresponding scripts in this folder.

## Local
Just run `./deepMNIST_local.sh`.

## Piz Daint
Assume USERNAME is your CSCS username.

1. Run `rsync deepMNIST.py DeepMNIST.ipynb deepMNIST_daint.sh USERNAME@daint.cscs.ch:MNIST/`.
2. Run `scp -r MNIST_data/ USERNAME@daint.cscs.ch:MNIST/`.
You need to copy `MNIST_data/` as no Internet connection is available at the GPU nodes (this folder is created when you run `deepMNIST_local.sh`).
3. Log into Piz Daint with `ssh USERNAME@daint.cscs.ch`.
4. Run `sbatch MNIST/deepMNIST_daint.sh`.

## AWS
Assume IP_INSTANCE is the IP address of your AWS EC2 instance.

1. Run `rsync deepMNIST.py DeepMNIST.ipynb deepMNIST_aws.sh ubuntu@IP_INSTANCE:MNIST/`.
2. Log into the instance with `ssh ubuntu@IP_INSTANCE`.
4. Run `./MNIST/deepMNIST_aws.sh`.