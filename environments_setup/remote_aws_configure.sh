#!/bin/bash

AWS_CONFIG_DIR=.aws
AWS_CONFIG_FILENAME=$AWS_CONFIG_DIR/config
PUB_IPS=../aws_public_ips.txt

NUM_HOSTS=`cat $PUB_IPS | wc -l`
for i in `seq 1 $NUM_HOSTS`; do
  ip=`sed "${i}q;d" $PUB_IPS`
  host=ubuntu@$ip
  ssh $host "mkdir -p $AWS_CONFIG_DIR"
  scp aws_config.txt $host:$AWS_CONFIG_FILENAME
done
