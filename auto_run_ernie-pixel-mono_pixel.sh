#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
step=1250

MODEL=pretrained_models/ernie-pixel-mono/checkpoint-${step}/
CKPT_NAME=ckpt-${step}
LOG_DIR=log/ft_ernie-pixel-mono/$CKPT_NAME

mkdir -p $LOG_DIR

for TASK in mnli qqp qnli sst2 cola mrpc stsb rte wnli
do
    echo "running ${TASK}..."
    bash run/ernie-pixel-mono/ft_ernie-pixel-mono_${TASK}_pixel.sh $MODEL > $LOG_DIR/ft_ernie-pixel-mono_${TASK}_pixel_${CKPT_NAME}.log 2>&1
done

echo "finished!"