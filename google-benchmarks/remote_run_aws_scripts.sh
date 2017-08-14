#!/bin/bash

OUTPUT_DIR=../outputs

# ./remote_run_dist_benchmarks_aws.sh TF_NUM_PS TF_NUM_WORKER parameter_server/distributed_replicated true/false NUM_GPU

for TF_NUM_WORKER in 1; do
  for TF_NUM_PS in 1; do
    for VARIABLE_UPDATE in parameter_server distributed_replicated; do
      for REAL_DATA in false; do
        for NUM_GPU in 8; do
          echo -e "\nRunning $TF_NUM_PS $TF_NUM_WORKER $VARIABLE_UPDATE $REAL_DATA $NUM_GPU"
          ./remote_run_dist_benchmarks_aws.sh $TF_NUM_PS $TF_NUM_WORKER \
           $VARIABLE_UPDATE $REAL_DATA $NUM_GPU > \
           $OUTPUT_DIR/$TF_NUM_PS\_$TF_NUM_WORKER\_$NUM_GPU\_$VARIABLE_UPDATE\_$REAL_DATA.out
        done
      done
    done
  done
done
