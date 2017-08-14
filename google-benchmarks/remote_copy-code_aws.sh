#!/bin/bash

PUB_IPS=../aws_public_ips.txt

# copy TF files to remote machines
TF_SCRIPT_FILES="tf_cnn_benchmarks util run_dist_tf_aws.sh"
NUM_HOSTS=`cat $PUB_IPS | wc -l`
for i in `seq 1 $NUM_HOSTS`; do
  ip=`sed "${i}q;d" $PUB_IPS`
  host=ubuntu@$ip
  scp -r -oStrictHostKeyChecking=no $TF_SCRIPT_FILES $host:;	
done < $PUB_IPS

