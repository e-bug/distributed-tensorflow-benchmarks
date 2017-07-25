#!/bin/bash

# set locale
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales

# NVIDIA requirements to run TensorFlow with GPU support
## CUDA Toolkit 8.0
sudo apt-get update
sudo apt install gcc
sudo apt-get install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
## Drivers
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt-get install nvidia-
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
## cuDNN v5.1 (copied from /apps/daint/UES/6.0.UP02/sandbox-ds/soures/cudnn-8.0-linux-x64-v5.1.tgz)
tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

## libcupti-dev library
sudo apt-get install libcupti-dev

# install packages
sudo apt install python
sudo apt install virtualenv

# create virtualenv
export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
cd $WORKON_HOME
virtualenv -p python3 tf-aws

# load virtualenv
source $WORKON_HOME/tf-aws/bin/activate

# install TensorFlow GPU
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp35-cp35m-linux_x86_64.whl

# install dependencies
pip install -r ../requirements.txt

# install aws-cli
pip install awscli

# deactivate virtualenv
deactivate

