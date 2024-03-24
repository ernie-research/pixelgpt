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

# =====================Settings========================
NUM_NODE=8
MASTER_POART=23450

MODALITY="image-text"

TASK="xnli"
MODEL=$1 # also works with "bert-base-cased", "roberta-base", etc.
RENDERING_BACKEND="pygame"  # Consider trying out both "pygame" and "pangocairo" to see which one works best
SEQ_LEN=256
BSZ=8
GRAD_ACCUM=None  # We found that higher batch sizes can sometimes make training more stable
LR=None
SEED=42
MAX_STEPS=None

WARMUP_STEPS=100
EVAL_STEPS=500
SAVE_STEPS=500

# early stopping
IS_EARLY_STOPPING=True
METRIC_FOR_BEST_MODEL="eval_accuracy"
EARLY_STOPPING_PATIENCE=8
GREATER_IS_BETTER=True


# === DEBUG ===
# RUN_NAME=test_preprocess-on-the-fly
# =============


for LR in 1e-5 3e-5 5e-5 1e-4
do
    for GRAD_ACCUM in 4 8
    do
        for MAX_STEPS in 30000
            do  
                RUN_NAME="experiment/cross_lingual/xnli/train_all/ernie-pixel-clm/${TASK}-$(basename ${MODEL})/${TASK}-$(basename ${MODEL})-${RENDERING_BACKEND}-${MODALITY}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${NUM_NODE}-${LR}-${MAX_STEPS}-${SEED}"

                python -m torch.distributed.launch --nproc_per_node=${NUM_NODE} --master_port=${MASTER_POART} scripts/training/run_ernie_xnli_translate_train_all_pair.py \
                --model_name_or_path=${MODEL} \
                --model_type=ernie-pixel \
                --processor_name="${MODEL},renderers/noto_renderer" \
                --modality=${MODALITY} \
                --task_name=${TASK} \
                --load_from_file=True \
                --data_file_dir=data/xnli \
                --rendering_backend=${RENDERING_BACKEND} \
                --remove_unused_columns=False \
                --max_steps=${MAX_STEPS} \
                --do_train \
                --do_eval \
                --do_predict \
                --max_seq_length=${SEQ_LEN} \
                --warmup_steps=${WARMUP_STEPS} \
                --per_device_train_batch_size=${BSZ} \
                --per_device_eval_batch_size=8 \
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
                --metric_for_best_model=${METRIC_FOR_BEST_MODEL} \
                --report_to=tensorboard \
                --log_predictions \
                --early_stopping=${IS_EARLY_STOPPING} \
                --early_stopping_patience=${EARLY_STOPPING_PATIENCE} \
                --greater_is_better=${GREATER_IS_BETTER} \
                --load_best_model_at_end=True \
                --seed=${SEED} \
                --bf16
            done
    done
done