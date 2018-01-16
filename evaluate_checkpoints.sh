#!/bin/bash

# version 0.1
# USAGE: $0 <PATH-TO-MODELS> <TMP-DIR>

TMPDIR=$(readlink -f $2)
MODELS=($(find $1 -type f -name *.meta))
touch $2/lastwatch

function evaluate_model {
    FULLPATH=$(readlink -f $1)
    MODELPREFIX=$(dirname $FULLPATH)"/"$(basename $FULLPATH .meta)

    echo model_checkpoint_path: "\"$MODELPREFIX\"" > $TMPDIR/checkpoint
    echo all_model_checkpoint_paths: "\"$MODELPREFIX\"" >> $TMPDIR/checkpoint 

    TRAIN_SCRIPT=tf_cnn_benchmarks/tf_cnn_benchmarks.py
    TRAIN_DIR=/scratch/snx3000/${USER}/cray-tf-traindir
    python3 $TRAIN_SCRIPT \
	--eval=True \
	--model=resnet50 \
	--num_batches=2000 \
	--batch_size=64 \
	--data_format=NCHW \
	--data_name=imagenet \
	--optimizer=sgd \
	--num_intra_threads=1 \
	--num_inter_threads=0 \
	--number_ml_comm_threads=2 \
	--summary_verbosity=1 \
	--eval_interval_secs=0 \
	--train_dir=$TMPDIR \
	--eval_dir=/scratch/snx3000/${USER}/cray-tf-evaldir \
	--data_dir=/scratch/snx3000/${USER}/TF
    
    echo '--- Evaluation of model '$MODELPREFIX' done. ---'
}

while true
do
    # Process models in MODELS list
    for i in "${!MODELS[@]}"; do
	evaluate_model ${MODELS[i]}
	unset 'MODELS[i]'
    done

    # Watch directory and add new models to MODELS list
    MODELS=($(find $1 -type f -name *.meta -cnewer $2/lastwatch))
    touch $2/lastwatch
    sleep 2s
done
