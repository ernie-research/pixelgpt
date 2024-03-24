#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
step=9625

MODEL=pretrained_models/ernie-pixel-clm/checkpoint-${step}/
CKPT_NAME=ckpt-${step}
LOG_DIR=log/ft_ernie-pixel-clm/$CKPT_NAME

mkdir -p $LOG_DIR

# for TASK in mnli qqp qnli sst2 cola mrpc stsb rte wnli
for TASK in mnli qqp
do
    echo "running ${TASK}..."
    bash run/ernie-pixel-clm/ft_ernie-pixel-clm_${TASK}_pair.sh $MODEL > $LOG_DIR/ft_ernie-pixel-clm_${TASK}_pair_${CKPT_NAME}.log 2>&1
done

echo "finished!"