#!/bin/bash

# get number of executions
if [ -z "$1" ]; then
  # default to 5
  num_scripts=5
else
  num_scripts=$1
fi

# launch scripts serially
for i in `seq 1 $num_scripts`;
do
  sbatch google-benchmarks/google-benchmarks_daint.sh
  
  # wait until it has finished
  running=`squeue -u cbugliar | wc -l`
  while [ $running -gt 1 ]; do
    sleep 20
    running=`squeue -u cbugliar | wc -l`
  done
done

# get results
for file in `ls google-benchmark_daint.*`; do
  result=`tail -2 $file | head -n 1 | cut -d' ' -f3`
  results=$results','$result
done
echo $results
