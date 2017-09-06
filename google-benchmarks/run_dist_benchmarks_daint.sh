#!/bin/bash

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: variable_update: parameter_server/distributed_replicated
#   $4: real_data: true/false
#   $5: num_executions: number of executions to run (default 5)

USERNAME=cbugliar
SCRIPT_DIR=$HOME/google-benchmarks
TF_DIR=$SCRATCH/tf
SCRIPT_NAME=google-benchmarks_dist_daint.sh
SCRIPT_ARGS="$1 $2 $3 $4"

# get number of executions
if [ -z "$5" ]; then
  # default to 5
  num_executions=5
else
  num_executions=$5
fi

# launch scripts serially
rm -f $TF_DIR/ps*
rm -f $TF_DIR/worker*
num_nodes=$2
for i in `seq 1 $num_executions`; do
  echo -e "\nRunning execution $i"

  sbatch $SCRIPT_DIR/$SCRIPT_NAME $SCRIPT_ARGS
  
  # wait until it has finished 
  ## only this job
#  running=`squeue -u $USERNAME | wc -l`
#  while [ $running -gt 1 ]; do
#    sleep 20
#    running=`squeue -u $USERNAME | wc -l`
#  done
  ## other jobs running as well
  sleep 1800 # 30 min

  # get average result for this execution
  exe_sum=0
  for file in `ls $TF_DIR/worker*`; do
    node_result=`tail -2 $file | head -n 1 | cut -d' ' -f3`
    exe_sum=`echo $exe_sum $node_result | awk '{print $1 + $2}'`
  done
  rm $TF_DIR/ps*
  rm $TF_DIR/worker*
  exe_result=`echo $exe_sum $num_nodes | awk '{print $1/$2}'`
  echo "exe_result: $exe_result"

  results=$results','$exe_result
done
results=`echo $results | tail --bytes +2`
echo $results
