#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ======== 输入参数 ========
step=51750
RENDER_MODE='gray'
# =========================


MODEL=pretrained_models/ernie-pixel-only/checkpoint-${step}/
CKPT_NAME=ckpt-${step}
LOG_DIR=log/ft_ernie-pixel-only/$CKPT_NAME

mkdir -p $LOG_DIR

# for TASK in mnli qqp qnli sst2 cola mrpc stsb rte wnli
for TASK in mnli
do
    echo "running ${TASK}..."
    bash run/render_mode/nlu_glue/ernie-pixel-only/ft_ernie-pixel-only_${TASK}.sh $MODEL $RENDER_MODE > $LOG_DIR/ft_ernie-pixel-only_${TASK}_${CKPT_NAME}.log 2>&1
done

echo "finished!"
