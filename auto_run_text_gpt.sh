#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
step=61000


MODEL=pretrained_models/text_gpt/checkpoint-${step}/
CKPT_NAME=ckpt-${step}
LOG_DIR=log/ft_text_gpt/$CKPT_NAME

mkdir -p $LOG_DIR

for TASK in qqp qnli sst2 cola mrpc stsb rte wnli
do
    echo "running ${TASK}..."
    bash run/text_gpt/ft_text_gpt_${TASK}.sh $MODEL > $LOG_DIR/ft_text_gpt_${TASK}_${CKPT_NAME}.log 2>&1
done

echo "finished!"