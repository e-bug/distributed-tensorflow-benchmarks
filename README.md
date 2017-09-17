# Distributed TensorFlow benchmarks

This repository provides code and results for benchmarking distributed training on Piz Daint (a Slurm-based supercomputer) and on Amazon EC2 instances.

While we provide scripts to easily submit TensorFlow applications across multiple nodes on any of these systems, our main contribution is the comparison of different distributed settings to achieve the best performance given the number of nodes and the system under study.
We use [Google's benchmarking scripts for TensorFlow](https://github.com/tensorflow/benchmarks/) to obtain the number of trained images per second.
We test clusters with both one and multiple GPUs per node, and with different inter-node networks.

Slides, report and an IPython notebook show the results for InceptionV3 in TensorFlow 1.1.0 (to compare our measurements with the ones available in the [TensorFlow benchmarks page](https://www.tensorflow.org/performance/benchmarks).
Each test is run 5 times and the times are averaged together. For each test, we pick the configuration that gives the best performance.
Following Google's approach, for each test, 10 warmup steps are done and then the next 100 steps are averaged.
All the measurements are available in [this spreadsheet](https://docs.google.com/spreadsheets/d/1u4LlBYWodwVQqO45LMiJbNRXzcGnmpnfX-vDyfFkgAA/edit?usp=sharing).

## Description of this repository

- `environments_setup/`: folder containing scripts to quickly setup a local workstation, Piz Daint and AWS EC2 instances to run TensorFlow.
- `run_jupyter_notebooks/`: folder containing scripts to easily start a Jupyter notebook on a local workstation, Piz Daint or an AWS EC2 instance.
- `distributed_tensorflow_launchers/`: folder containing scripts to promptly launch a TensorFlow application across multiple processes on a local workstation, Piz Daint or AWS EC2 instances. In particular, a script takes care of starting Parameter Servers and Workers on the nodes allocated by Slurm.
- `MNIST/`: folder containing TensorFlow's deep MNIST tutorial and two variations: a GPU-enhanced version that allows to specify data formats (NCHW or NHWC) and a distributed version of this one.
- `google-benchmarks/`: folder containing scripts to run Google's benchmarking code on different systems.
- `report/`: folder containing the report for the summer internship at CSCS. 
- `presentation/`: folder containing the slides for the end-of-internship seminar given at CSCS.
