#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

# ==== 设置ckpt step ====
step=12500

MODEL=pretrained_models/ernie-pixel-mono/checkpoint-${step}/
CKPT_NAME=ckpt-${step}

# train-all ernie-pixel-mono text
# LOG_DIR=log/cross_lingual/xnli/train_all/ernie-pixel-mono/$CKPT_NAME
# mkdir -p $LOG_DIR
# bash run/cross_lingual/xnli/train_all/ernie-pixel-mono/ft_ernie-pixel_mono_xnli_text.sh $MODEL > $LOG_DIR/${CKPT_NAME}_text.log 2>&1
# sleep 60

# train-en ernie-pixel-mono text
LOG_DIR=log/cross_lingual/xnli/train_en/ernie-pixel-mono/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/cross_lingual/xnli/train_en/ernie-pixel-mono/ft_ernie-pixel_mono_xnli_text.sh $MODEL > $LOG_DIR/${CKPT_NAME}_text.log 2>&1
sleep 60

# train-all ernie-pixel-mono image
LOG_DIR=log/cross_lingual/xnli/train_all/ernie-pixel-mono/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/cross_lingual/xnli/train_all/ernie-pixel-mono/ft_ernie-pixel_mono_xnli_image.sh $MODEL > $LOG_DIR/${CKPT_NAME}_image.log 2>&1
sleep 60

# train-en ernie-pixel-mono image
LOG_DIR=log/cross_lingual/xnli/train_en/ernie-pixel-mono/$CKPT_NAME
mkdir -p $LOG_DIR
bash run/cross_lingual/xnli/train_en/ernie-pixel-mono/ft_ernie-pixel_mono_xnli_image.sh $MODEL > $LOG_DIR/${CKPT_NAME}_image.log 2>&1
sleep 60


echo "finished!"