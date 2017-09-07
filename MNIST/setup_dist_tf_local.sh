#!/bin/bash

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers


# set TensorFlow distributed parameters
TF_NUM_PS=$1
TF_NUM_WORKER=$2

PS_START_PORT=2230
PS_END_PORT=$((PS_START_PORT + TF_NUM_PS -1))
TF_PS_HOSTS=`printf 'localhost:%s,' $(seq -s' ' $PS_START_PORT $PS_END_PORT)`
TF_PS_HOSTS=$(echo "${TF_PS_HOSTS}" | head --bytes -2) # XXX -- assume 1 PS per PS host

WORKER_START_PORT=2220
WORKER_END_PORT=$((WORKER_START_PORT + TF_NUM_WORKER -1))
TF_WORKER_HOSTS=`printf 'localhost:%s,' $(seq -s' ' $WORKER_START_PORT $WORKER_END_PORT)`
TF_WORKER_HOSTS=$(echo "${TF_WORKER_HOSTS}" | head --bytes -2) # XXX -- assume 1 WORKER per WORKER host

RUN_TF_SCRIPT=run_dist_tf_local.sh

# start PSs
echo "starting PSs..."
for running_ps in `seq 1 $TF_NUM_PS`; do
  ps_idx=$(($running_ps - 1))
  gnome-terminal -x sh -c "./$RUN_TF_SCRIPT ps $ps_idx $TF_PS_HOSTS $TF_WORKER_HOSTS; bash"
done

# start WORKERs
echo "starting WORKERs..."
running_w=0
for running_w in `seq 1 $((TF_NUM_WORKER - 1))`; do
  w_idx=$(($running_w - 1))
  gnome-terminal -x sh -c "./$RUN_TF_SCRIPT worker $w_idx $TF_PS_HOSTS $TF_WORKER_HOSTS; bash"
done
w_idx=$running_w
./$RUN_TF_SCRIPT worker $w_idx $TF_PS_HOSTS $TF_WORKER_HOSTS
