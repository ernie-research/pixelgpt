#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
step=12500 # 指定你的ckpt步数
PT_TYPE=pixel_gpt # 指定你的预训练类型
# =======================


# pretrained model path
MODEL=pretrained_models/$PT_TYPE/checkpoint-${step}/

CKPT_NAME=ckpt-${step}

# ===== train-all; ft image =====
LOG_DIR=log/cross_lingual/xnli/train_all/$PT_TYPE/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/cross_lingual/xnli/train_all/$PT_TYPE/ft_${PT_TYPE}_xnli.sh $MODEL > $LOG_DIR/${CKPT_NAME}.log 2>&1
sleep 60

# ===== train-en; ft image =====
LOG_DIR=log/cross_lingual/xnli/train_en/$PT_TYPE/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/cross_lingual/xnli/train_en/$PT_TYPE/ft_${PT_TYPE}_xnli.sh $MODEL > $LOG_DIR/${CKPT_NAME}.log 2>&1
sleep 60


echo "finished!"