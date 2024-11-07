#!/bin/bash

LR=5e-4
EPOCH=50
SEED=42

# Define common parameters
COMMON_PARAMS="--dataset_name nnheui/shortestpath_50_15_2_4 \
               --remove_unused_columns False \
               --tokenizer_name gpt --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --preprocessing_num_workers 4 \
               --learning_rate $LR --num_train_epochs $EPOCH --eval_steps 500 --evaluation_strategy no \
               --save_total_limit 10 --save_step 0.1 --report_to none --seed $SEED --bf16 --logging_steps 500"

# Function to run an experiment
run_experiment() {
    local gpu_id=$1
    local model_config=$2
    local output_dir=$3
    local additional_params=$4
    local eval_only=${5:-0}

    # add --do_eval if 1 else --do_train
    local exp_params=$COMMON_PARAMS 
    if [ $eval_only -eq 1 ]; then
        exp_params="$exp_params --do_eval"
    else
        exp_params="$exp_params --do_train"
    fi

    accelerate launch --gpu_ids $gpu_id --mixed_precision=bf16 main.py $exp_params \
                $model_config \
                --output_dir $output_dir \
                $additional_params
}

# Run experiments
run_experiment 3 "--config_name model_configs/pythia_small.json" "outputs/pretrain" "--objective step_eval"