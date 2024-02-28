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
export TASK="rte"
export MODEL="pretrained_models/ernie-clm-base/checkpoint-61000" # also works with "bert-base-cased", "roberta-base", etc.
export RENDERING_BACKEND="pygame"  # Consider trying out both "pygame" and "pangocairo" to see which one works best
export SEQ_LEN=1024
export BSZ=4
export GRAD_ACCUM=8  # We found that higher batch sizes can sometimes make training more stable
export LR=3e-5
export SEED=42
export EPOCHS=3

export RUN_NAME="ernie-clm-base-${TASK}-$(basename ${MODEL})-${RENDERING_BACKEND}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${EPOCHS}-${SEED}"

# === DEBUG ===
# export RUN_NAME=test
# =============

# python scripts/training/run_ernie-pixel_glue.py \
#   --max_train_samples=32 \
#   --max_eval_samples=32 \
#   --max_predict_samples=32 \
python scripts/training/run_ernie-pixel_glue.py \
  --model_name_or_path=${MODEL} \
  --task_name=${TASK} \
  --rendering_backend=${RENDERING_BACKEND} \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length=${SEQ_LEN} \
  --num_train_epochs=${EPOCHS} \
  --early_stopping \
  --early_stopping_patience=5 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=10 \
  --run_name=${RUN_NAME} \
  --output_dir=${RUN_NAME} \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=10 \
  --evaluation_strategy=steps \
  --eval_steps=100 \
  --save_strategy=steps \
  --save_steps=100 \
  --report_to=tensorboard \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_accuracy" \
  --fp16 \
  --seed=${SEED}