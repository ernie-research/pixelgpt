#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
# step=25000 # 指定你的ckpt步数
# PT_TYPE=ernie-clm-base # 指定你的预训练类型
# =======================


for step in 61000 25000 12500 10000 5000 2500
do
PT_TYPE=text_gpt # 指定你的预训练类型


# pretrained model path
MODEL=pretrained_models/$PT_TYPE/checkpoint-${step}/

CKPT_NAME=ckpt-${step}

# ===== train-all; ft text =====
LOG_DIR=log/cross_lingual/xnli/train_all/$PT_TYPE/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/cross_lingual/xnli/train_all/$PT_TYPE/ft_${PT_TYPE}_xnli.sh $MODEL > $LOG_DIR/${CKPT_NAME}.log 2>&1
sleep 60

# ===== train-en; ft text =====
# LOG_DIR=log/cross_lingual/xnli/train_en/$PT_TYPE/$CKPT_NAME
# mkdir -p $LOG_DIR
# bash run/cross_lingual/xnli/train_en/$PT_TYPE/ft_${PT_TYPE}_xnli.sh $MODEL > $LOG_DIR/${CKPT_NAME}.log 2>&1
# sleep 60


echo "finished!"
done
