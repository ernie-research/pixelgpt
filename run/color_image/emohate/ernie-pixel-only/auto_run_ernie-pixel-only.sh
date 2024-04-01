#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ======== 输入参数 ========
step=51750
RENDER_MODE="gray"
# =========================


MODEL=pretrained_models/ernie-pixel-only/checkpoint-${step}/
CKPT_NAME=ckpt-${step}
LOG_DIR=log/color_image/emohate/$RENDER_MODE/ernie-pixel-only/$CKPT_NAME
mkdir -p $LOG_DIR

mkdir -p $LOG_DIR

echo "running emohate..."
bash run/color_image/emohate/ernie-pixel-only/ft_ernie-pixel-only_emohate.sh $RENDER_MODE $MODEL > $LOG_DIR/ft_ernie-pixel-only_${TASK}_${CKPT_NAME}.log 2>&1

echo "finished!"
