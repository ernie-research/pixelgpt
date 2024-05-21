#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
step=9625

MODEL=pretrained_models/dual_gpt/checkpoint-${step}/
CKPT_NAME=ckpt-${step}
LOG_DIR=log/ft_dual_gpt/$CKPT_NAME

mkdir -p $LOG_DIR

for TASK in mnli qqp qnli sst2 cola mrpc stsb rte wnli
do
    echo "running ${TASK}..."
    bash run/dual_gpt/ft_dual_gpt_${TASK}_pair.sh $MODEL > $LOG_DIR/ft_dual_gpt_${TASK}_pair_${CKPT_NAME}.log 2>&1
done

echo "finished!"