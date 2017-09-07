# MNIST tutorials

## Running TensorFlow's MNIST tutorial

You can run [TensorFlow's MNIST tutorial](https://www.tensorflow.org/get_started/mnist/pros) in two ways:

- in an interactive way through the `DeepMNIST.ipynb` notebook;
- with a script that calls `deepMNIST.py`.

In the former case, just launch a Jupyter notebook (as described in [`run_jupyter_notebooks/`](../run_jupyter_notebooks)) and browse to the MNIST directory.
For the latter, just run the corresponding scripts in this folder.

### Local
Just run `./deepMNIST_local.sh`.

### Piz Daint
Assume USERNAME is your CSCS username.

1. Run `rsync deepMNIST.py DeepMNIST.ipynb deepMNIST_daint.sh USERNAME@daint.cscs.ch:MNIST/`.
2. Run `scp -r MNIST_data/ USERNAME@daint.cscs.ch:MNIST/`.
You need to copy `MNIST_data/` as no Internet connection is available at the GPU nodes (this folder is created when you run `deepMNIST_local.sh`).
3. Log into Piz Daint with `ssh USERNAME@daint.cscs.ch`.
4. Run `sbatch MNIST/deepMNIST_daint.sh`.

### AWS
Assume IP_INSTANCE is the IP address of your AWS EC2 instance.

1. Run `rsync deepMNIST.py DeepMNIST.ipynb deepMNIST_aws.sh ubuntu@IP_INSTANCE:MNIST/`.
2. Log into the instance with `ssh ubuntu@IP_INSTANCE`.
4. Run `./MNIST/deepMNIST_aws.sh`.


## GPU-enhanced MNIST
We provide a version of the MNIST tutorial that makes use of the NCHW data format to run efficiently on GPUs. <br>
The aim of this version is to show how to write a TensorFlow application that allows to use the NCHW data format on GPUs when input data cosists of images.
Having only two convolutional layers in the MNIST network, the performance gain is marginal.

- `deepMNIST_gpu.py`: GPU-enhanced version of TensorFlow's MNIST tutorial that allows to use the NCHW data format.

### Local
Just run `./deepMNIST_gpu_local.sh`.

### Piz Daint
Assume USERNAME is your CSCS username.

1. Run `rsync deepMNIST_gpu.py deepMNIST_gpu_daint.sh USERNAME@daint.cscs.ch:MNIST/`.
2. Run `scp -r MNIST_data/ USERNAME@daint.cscs.ch:MNIST/`.
You need to copy `MNIST_data/` as no Internet connection is available at the GPU nodes (this folder is created when you run `deepMNIST_gpu_local.sh`).
3. Log into Piz Daint with `ssh USERNAME@daint.cscs.ch`.
4. Run `sbatch MNIST/deepMNIST_gpu_daint.sh`.


## Distributed GPU-enhanced MNIST
We provide a version of the MNIST tutorial that creates a cluster of TensorFlow servers and trains the MNIST network across that cluster. <br>
The aim of this version is to show how to write a distributed TensorFlow application starting from an existing single-machine TensorFlow application (GPU-enhanced MNIST, here).

- `dist_deepMNIST_gpu.py`: distributed version of the GPU-enhanced TensorFlow's MNIST tutorial presented above.

### Local
1. Update `run_dist_tf_local.sh` with your settings.
2. Run `./setup_dist_tf_local.sh`. Arguments: number of Parameter Servers and number of Workers.

### Piz Daint
Assume USERNAME is your CSCS username.

1. Run `rsync dist_deepMNIST_gpu.py run_dist_tf_daint.sh setup_dist_tf_daint.sh USERNAME@daint.cscs.ch:MNIST/`.
2. Run `scp -r MNIST_data/ USERNAME@daint.cscs.ch:MNIST/`.
You need to copy `MNIST_data/` as no Internet connection is available at the GPU nodes (this folder is created when you run `deepMNIST_gpu_local.sh`).
3. Log into Piz Daint with `ssh USERNAME@daint.cscs.ch`.
4. Update `setup_dist_tf_daint.sh` with your settings.
5. Run `sbatch MNIST/setup_dist_tf_daint.sh`. Arguments: number of Parameter Servers and number of Workers.