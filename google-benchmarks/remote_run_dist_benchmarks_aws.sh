#!/bin/bash

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: variable_update: parameter_server/distributed_replicated
#   $4: real_data: true/false
#   $5: NUM_GPU: number of GPUs per worker
#   $6: num_executions: number of executions to run (default 5)

# TF script filenames
SCRIPT_DIR=~/aws/code
SCRIPT_NAME=remote_google-benchmarks_dist_aws.sh
SCRIPT_ARGS="$1 $2 $3 $4 $5"
OUTPUT_FILENAME=screenlog.0

# IP files
PRI_IPS=../aws_private_ips.txt
PUB_IPS=../aws_public_ips.txt

AWS_PRIVATE_KEY=~/.ssh/aws

# get number of executions
if [ -z "$6" ]; then
  # default to 5
  num_executions=5
else
  num_executions=$6
fi

# max execution time
MAX_TIME=720 # 12min

# launch scripts serially
num_nodes=$2
for i in `seq 1 $num_executions`; do
  echo -e "\nRunning execution $i"
  exe_sum=0

  # clean nodes
  for running_w in `seq 1 $2`; do
    ip=`sed "${running_w}q;d" $PUB_IPS`
    host=ubuntu@$ip
    ssh -oStrictHostKeyChecking=no -i $AWS_PRIVATE_KEY $host "pkill screen; \
        rm -f $OUTPUT_FILENAME"
  done

  # run distributed TF script
  cd $SCRIPT_DIR
  ./$SCRIPT_NAME $SCRIPT_ARGS
  
  sleep 60

  # wait until they have finished or MAX_TIME is reached
  end_time=$(($SECONDS+$MAX_TIME))
  stop_exe=false
  for running_w in `seq 1 $2`; do
    ip=`sed "${running_w}q;d" $PUB_IPS`
    host=ubuntu@$ip
    scp -i $AWS_PRIVATE_KEY $host:$OUTPUT_FILENAME .
    while [ `tail -2 $OUTPUT_FILENAME | head -1 | cut -d ' ' -f 1` != "total" ]
    do
      if [ $SECONDS -gt $end_time ]; then
        # do not wait anymore for this WORKER
        stop_exe=true
        break
      fi
      sleep 60
      scp -i $AWS_PRIVATE_KEY $host:$OUTPUT_FILENAME .
    done
    if [ $stop_exe = "true" ]; then
      # do not wait for the other WORKERs
      break
    fi
    node_result=`tail -2 $OUTPUT_FILENAME | head -n 1 | cut -d ' ' -f 3 | 
                 head --bytes -1`
    exe_sum=`awk "BEGIN {print $exe_sum+$node_result}"`
    rm -f $OUTPUT_FILENAME
  done

  # get average result for this execution
  if [ $stop_exe = "true" ]; then
    echo "exe_result: Wall reached"
    continue
  fi
  exe_result=`bc <<< "scale=3; $exe_sum/$num_nodes"`
  echo "exe_result: $exe_result"

  results=$results','$exe_result
done
results=`echo $results | tail --bytes +2`
echo $results
