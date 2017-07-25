#!/bin/bash

#SBATCH --job-name=google_benchmark
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --mem=60GB
#SBATCH --output=google-benchmark_daint.%j.log

# load modules
module load daint-gpu
module load TensorFlow/1.0.0-CrayGNU-2016.11-cuda-8.0-Python-3.5.2

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

# train inception
cd ~/google-benchmarks
srun python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--num_gpus=1 \
--batch_size=64 \
--data_format=NCHW \
--use_nccl=True \
--variable_update=parameter_server \
--local_parameter_device=gpu \
--optimizer=sgd \
--model=inception3 \
--data_name=imagenet \
#--data_dir=/scratch/snx3000/maximem/deeplearnpackages/ImageNet/TF/

# deactivate virtualenv
deactivate

