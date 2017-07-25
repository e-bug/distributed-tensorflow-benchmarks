#!/bin/bash

# get number of executions
if [ -z "$1" ]; then
  # default to 5
  num_scripts=5
else
  num_scripts=$1
fi

# launch parallel scripts
for i in `seq 1 $num_scripts`;
do
  sbatch google-benchmarks/google-benchmarks_daint.sh
done

# wait until all have finished
running=`squeue -u cbugliar | wc -l`
while [ $running -gt 1 ]; do
  sleep 20
  running=`squeue -u cbugliar | wc -l`
done

# get results
for file in `ls google-benchmark_daint.*`; do
  result=`tail -2 $file | head -n 1 | cut -d' ' -f3`
  results=$results','$result
done
echo $results
