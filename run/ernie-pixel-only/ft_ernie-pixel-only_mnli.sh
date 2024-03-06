#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# Note on GLUE: 
# We found that for some of the tasks (e.g. MNLI), PIXEL can get stuck in a bad local optimum
# A clear indicator of this is when the training loss is not decreasing substantially within the first 1k-3k steps
# If this happens, you can tweak the learning rate slightly, increase the batch size,
# change rendering backends, or often even just the random seed
# We are still trying to find the optimal training recipe for PIXEL on these tasks,
# the recipes used in the paper may not be the best ones out there

# Settings
TASK="mnli"
MODEL="pretrained_models/ernie-pixel-only/checkpoint-27500/" # also works with "bert-base-cased", "roberta-base", etc.
RENDERING_BACKEND="pygame"  # Consider trying out both "pygame" and "pangocairo" to see which one works best
SEQ_LEN=768
BSZ=4
GRAD_ACCUM=None  # We found that higher batch sizes can sometimes make training more stable
LR=None
SEED=42
MAX_STEPS=None

WARMUP_STEPS=100
EVAL_STEPS=500
SAVE_STEPS=500



# === DEBUG ===
# RUN_NAME=test_preprocess-on-the-fly
# =============

for LR in 5e-5
do
    for GRAD_ACCUM in 2
    do
        for MAX_STEPS in 15000
            do
                RUN_NAME="ernie-pixel-only-${TASK}-$(basename ${MODEL})-${RENDERING_BACKEND}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${MAX_STEPS}-${SEED}"

                python -m torch.distributed.launch --nproc_per_node=8 scripts/training/run_ernie-pixel_glue.py \
                --model_name_or_path=${MODEL} \
                --model_type=ernie-pixel \
                --processor_name=renderers/noto_renderer \
                --task_name=${TASK} \
                --load_from_file=True \
                --train_file=/root/paddlejob/workspace/env_run/liuqingyi01/pixel_data/mnli-train/part-00000.gz \
                --validation_file=/root/paddlejob/workspace/env_run/liuqingyi01/pixel_data/mnli-validation_mismatched/part-00000.gz \
                --test_file=/root/paddlejob/workspace/env_run/liuqingyi01/pixel_data/mnli-test_mismatched/part-00000.gz \
                --validation_matched_file=/root/paddlejob/workspace/env_run/liuqingyi01/pixel_data/mnli-validation_matched/part-00000.gz \
                --test_matched_file=/root/paddlejob/workspace/env_run/liuqingyi01/pixel_data/mnli-test_matched/part-00000.gz \
                --rendering_backend=${RENDERING_BACKEND} \
                --remove_unused_columns=False \
                --max_steps=${MAX_STEPS} \
                --do_train \
                --do_eval \
                --do_predict \
                --max_seq_length=${SEQ_LEN} \
                --early_stopping=False \
                --warmup_steps=${WARMUP_STEPS} \
                --per_device_train_batch_size=${BSZ} \
                --gradient_accumulation_steps=${GRAD_ACCUM} \
                --learning_rate=${LR} \
                --run_name=${RUN_NAME} \
                --output_dir=${RUN_NAME} \
                --overwrite_output_dir \
                --overwrite_cache \
                --logging_strategy=steps \
                --logging_steps=1 \
                --evaluation_strategy=steps \
                --eval_steps=${EVAL_STEPS} \
                --save_strategy=steps \
                --save_steps=${SAVE_STEPS} \
                --save_total_limit=1 \
                --report_to=tensorboard \
                --log_predictions \
                --bf16 \
                --load_best_model_at_end=True \
                --seed=${SEED}
            done
    done
done



