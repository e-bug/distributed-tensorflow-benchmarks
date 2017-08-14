#!/bin/bash

PUB_IPS=../aws_public_ips.txt

NUM_HOSTS=`cat $PUB_IPS | wc -l`
for i in `seq 1 $NUM_HOSTS`; do
  ip=`sed "${i}q;d" $PUB_IPS`
  host=ubuntu@$ip
  ssh -oStrictHostKeyChecking=no $host 'screen -S data -d -m 
      source Envs/tf-aws/bin/activate; 
      aws s3 cp --recursive s3://tfcscs/imagenet ~/imagenet' &
done
