#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
step=51750 # 指定你的ckpt步数
PT_TYPE=ernie-pixel-only # 指定你的预训练类型
# =======================


# pretrained model path
MODEL=pretrained_models/$PT_TYPE/checkpoint-${step}/

CKPT_NAME=ckpt-${step}

# ===== font =====
### ===== notoserif =====
for FONT in notoserif_renderer journal_renderer
do
    LOG_DIR=log/font/cola/ernie-pixel-only/$FONT/$PT_TYPE/$CKPT_NAME
    mkdir -p $LOG_DIR
    bash run/font/cola/ernie-pixel-only/ft_${PT_TYPE}_cola_font.sh $FONT $MODEL > $LOG_DIR/${CKPT_NAME}.log 2>&1
    sleep 60
done


echo "finished!"