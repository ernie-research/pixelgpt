#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
step=5000

MODEL=pretrained_models/mono_gpt/checkpoint-${step}/
CKPT_NAME=ckpt-${step}
LOG_DIR=log/ft_mono_gpt/$CKPT_NAME

mkdir -p $LOG_DIR

for TASK in mnli qqp qnli sst2 cola mrpc stsb rte wnli
do
    echo "running ${TASK}..."
    bash run/mono_gpt/ft_mono_gpt_${TASK}_pixel.sh $MODEL > $LOG_DIR/ft_mono_gpt_${TASK}_pixel_${CKPT_NAME}.log 2>&1
done

echo "finished!"