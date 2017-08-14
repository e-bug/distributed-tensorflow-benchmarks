#!/bin/bash

PUB_IPS=../aws_public_ips.txt

NUM_HOSTS=`cat $PUB_IPS | wc -l`
num_ok=0
bad_ips=

# Run tests
for i in `seq 1 $NUM_HOSTS`; do
  ip=`sed "${i}q;d" $PUB_IPS`
  host=ubuntu@$ip
  ssh -oStrictHostKeyChecking=no $host '~/test_tf_aws.sh &> test_tf_aws.out' &
done

# Make sure tests complete
sleep 60

# Verify tests
for i in `seq 1 $NUM_HOSTS`; do
  ip=`sed "${i}q;d" $PUB_IPS`
  host=ubuntu@$ip
  scp $host:test_tf_aws.out .
  if [[ `tail -1 test_tf_aws.out` = "b'Hello, TensorFlow'" ]]; then
    num_ok=$(($num_ok + 1))
  else
    bad_ips=$bad_ips','$ip
  fi
  ssh $host 'rm test_tf_aws.out' &
done
rm test_tf_aws.out

# Print result
if [ $num_ok = $NUM_HOSTS ]; then
  echo "All node setup correctly"
else
  echo "Unsucceful setup for IPs: $bad_ips"
fi
