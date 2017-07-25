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
  filename='aws_inception3_'`date +%b_%d_%Y_%H_%M_%S`'.log'
  ./google-benchmarks/google-benchmarks_aws.sh > $filename
  result=`tail -2 $filename | head -n 1 | cut -d' ' -f3`
  results=$results','$result
done
echo $results
