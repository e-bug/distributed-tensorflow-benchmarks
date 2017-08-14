#!/bin/bash

PUB_IPS=../aws_public_ips.txt

NUM_HOSTS=`cat $PUB_IPS | wc -l`
for i in `seq 1 $NUM_HOSTS`; do
  ip=`sed "${i}q;d" $PUB_IPS`
  host=ubuntu@$ip
  scp -oStrictHostKeyChecking=no requirements.txt cudnn-8.0-linux-x64-v5.1.tgz \
      setup_virtenv_aws.sh test_tf_aws.sh $host: 
  ssh $host 'screen -S setup -d -m ~/setup_virtenv_aws.sh' &
done
