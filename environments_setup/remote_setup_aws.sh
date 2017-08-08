#!/bin/bash

PUB_IPS=../aws_public_ips.txt
while read line
do
  host=ubuntu@$line
  scp -oStrictHostKeyChecking=no requirements.txt cudnn-8.0-linux-x64-v5.1.tgz \
      setup_virtenv_aws.sh test_tf_aws.sh $host: 
  ssh -oStrictHostKeyChecking=no $host 'screen -S setup -d -m ~/setup_virtenv_aws.sh' &
done < $PUB_IPS
