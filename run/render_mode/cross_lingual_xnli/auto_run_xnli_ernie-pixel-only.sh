#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
step=12500 # 指定你的ckpt步数
PT_TYPE=ernie-pixel-only # 指定你的预训练类型
# =======================


# pretrained model path
MODEL=pretrained_models/$PT_TYPE/checkpoint-${step}/

CKPT_NAME=ckpt-${step}

# ===== train-all; ft image =====
### ===== binary =====
LOG_DIR=log/render_mode/cross_lingual_xnli/train_all/$PT_TYPE/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/render_mode/cross_lingual_xnli/train_all/ft_${PT_TYPE}_xnli.sh $MODEL "binary" > $LOG_DIR/${CKPT_NAME}.log 2>&1
sleep 60
### ===== grayscale =====
LOG_DIR=log/render_mode/cross_lingual_xnli/train_all/$PT_TYPE/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/render_mode/cross_lingual_xnli/train_all/ft_${PT_TYPE}_xnli.sh $MODEL "gray" > $LOG_DIR/${CKPT_NAME}.log 2>&1
sleep 60

# ===== train-en; ft image =====
### ===== binary =====
LOG_DIR=log/render_mode/cross_lingual_xnli/train_en/$PT_TYPE/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/render_mode/cross_lingual_xnli/train_en/ft_${PT_TYPE}_xnli.sh $MODEL "binary" > $LOG_DIR/${CKPT_NAME}.log 2>&1
sleep 60
### ===== grayscale =====
LOG_DIR=log/render_mode/cross_lingual_xnli/train_en/$PT_TYPE/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/render_mode/cross_lingual_xnli/train_en/ft_${PT_TYPE}_xnli.sh $MODEL "gray" > $LOG_DIR/${CKPT_NAME}.log 2>&1
sleep 60


echo "finished!"