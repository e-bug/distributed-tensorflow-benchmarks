#!/bin/bash

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: NUM_GPU: number of GPUs per worker

# IP files
PRI_IPS=../aws_private_ips.txt
PUB_IPS=../aws_public_ips.txt

AWS_PRIVATE_KEY=~/.ssh/aws.pem

# set TensorFlow distributed parameters
TF_NUM_PS=$1
TF_NUM_WORKER=$2

TF_PS_HOSTS=$(head -$TF_NUM_PS $PRI_IPS | tr -s '\n' ',')
TF_PS_HOSTS=$(echo "${TF_PS_HOSTS//,/:2230,}" | head --bytes -2) # XXX -- assume 1 PS per PS host
TF_WORKER_HOSTS=$(head -$TF_NUM_WORKER $PRI_IPS | tr -s '\n' ',')
TF_WORKER_HOSTS=$(echo "${TF_WORKER_HOSTS//,/:2220,}" | head --bytes -2) # XXX -- assume 1 WORKER per WORKER host

RUN_TF_SCRIPT=$HOME/run_dist_tf_aws.sh

# start WORKERs
echo "starting WORKERs..."
for running_w in `seq 1 $TF_NUM_WORKER`; do
  ip=`sed "${running_w}q;d" $PUB_IPS`
  host=ubuntu@$ip
  w_idx=$(($running_w - 1))
  ssh -oStrictHostKeyChecking=no -i $AWS_PRIVATE_KEY $host "screen -L -S \
      w_$w_idx -d -m $RUN_TF_SCRIPT worker $w_idx \
      $TF_PS_HOSTS $TF_WORKER_HOSTS $3" &
done

sleep 60

# start PSs
echo "starting PSs..."
for running_ps in `seq 1 $TF_NUM_PS`; do
  ip=`sed "${running_ps}q;d" $PUB_IPS`
  host=ubuntu@$ip
  ps_idx=$(($running_ps - 1))
  ssh -oStrictHostKeyChecking=no -i $AWS_PRIVATE_KEY $host "screen -S \
      ps_$ps_idx -d -m $RUN_TF_SCRIPT ps $ps_idx \
      $TF_PS_HOSTS $TF_WORKER_HOSTS $3" &
done

