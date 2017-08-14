#!/bin/bash

OUTPUT_DIR=outputs

# ./run_dist_benchmarks_daint.sh TF_NUM_PS TF_NUM_WORKER parameter_server/distributed_replicated true/false

for TF_NUM_WORKER in 32; do
  for TF_NUM_PS in 2 4; do
    for VARIABLE_UPDATE in parameter_server; do
      for REAL_DATA in false; do
        echo -e "\nRunning $TF_NUM_PS $TF_NUM_WORKER $VARIABLE_UPDATE $REAL_DATA"
        ./run_dist_benchmarks_daint.sh $TF_NUM_PS $TF_NUM_WORKER \
         $VARIABLE_UPDATE $REAL_DATA > \
         $OUTPUT_DIR/$TF_NUM_PS\_$TF_NUM_WORKER\_$VARIABLE_UPDATE\_$REAL_DATA.out
      done
    done
  done
done

