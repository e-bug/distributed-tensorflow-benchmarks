#!/bin/bash

USERNAME=cbugliar
SCRIPT_DIR=$HOME/google-benchmarks
SCRIPT_NAME=google-benchmarks_daint.sh
SLURM_OUTPUTS=benchmark_daint.*

# get number of executions
if [ -z "$1" ]; then
  # default to 5
  num_executions=5
else
  num_executions=$1
fi

# launch scripts
rm -f $SLURM_OUTPUTS
for i in `seq 1 $num_executions`;
do
  sbatch $SCRIPT_DIR/$SCRIPT_NAME
  
  # wait until it has finished
  running=`squeue -u $USERNAME | wc -l`
  while [ $running -gt 1 ]; do
    sleep 20
    running=`squeue -u $USERNAME | wc -l`
  done
done

# get results
for file in `ls $SLURM_OUTPUTS`; do
  result=`tail -2 $file | head -n 1 | cut -d' ' -f3`
  results=$results','$result
done
rm $SLURM_OUTPUTS
echo $results
