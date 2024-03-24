#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

echo "sleeping..."
sleep 1800

# ==== 设置ckpt step ====
step=12500 # 指定你的ckpt步数
PT_TYPE=ernie-pixel-mono # 指定你的预训练类型
# =======================


# pretrained model path
MODEL=pretrained_models/$PT_TYPE/checkpoint-${step}/

CKPT_NAME=ckpt-${step}

# ===== train-all; ft text =====
# LOG_DIR=log/cross_lingual/xnli/train_all/$PT_TYPE/$CKPT_NAME
# mkdir -p $LOG_DIR
# bash run/cross_lingual/xnli/train_all/$PT_TYPE/ft_${PT_TYPE}_xnli_text.sh $MODEL > $LOG_DIR/${CKPT_NAME}_text.log 2>&1
# sleep 60

# ===== train-en; ft text =====
# LOG_DIR=log/cross_lingual/xnli/train_en/$PT_TYPE/$CKPT_NAME
# mkdir -p $LOG_DIR
# bash run/cross_lingual/xnli/train_en/$PT_TYPE/ft_${PT_TYPE}_xnli_text.sh $MODEL > $LOG_DIR/${CKPT_NAME}_text.log 2>&1
# sleep 60

# ===== train-all; ft image =====
LOG_DIR=log/cross_lingual/xnli/train_all/$PT_TYPE/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/cross_lingual/xnli/train_all/$PT_TYPE/ft_${PT_TYPE}_xnli_image.sh $MODEL > $LOG_DIR/${CKPT_NAME}_image.log 2>&1
sleep 60

# ===== train-en; ft image =====
# LOG_DIR=log/cross_lingual/xnli/train_en/$PT_TYPE/$CKPT_NAME
# mkdir -p $LOG_DIR
# bash run/cross_lingual/xnli/train_en/$PT_TYPE/ft_${PT_TYPE}_xnli_image.sh $MODEL > $LOG_DIR/${CKPT_NAME}_image.log 2>&1
# sleep 60


echo "finished!"