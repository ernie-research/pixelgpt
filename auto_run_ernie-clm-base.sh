#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

CKPT_NAME=ckpt-25000
LOG_DIR=log/ft_ernie-clm-base/$CKPT_NAME
mkdir -p $LOG_DIR

echo "running MNLI..."
TASK="mnli"
bash run/ernie-clm-base/ft_ernie-clm-base_${TASK}.sh > $LOG_DIR/ft_ernie-clm-base_${TASK}_${CKPT_NAME}.log 2>&1

echo "running QQP..."
TASK="qqp"
bash run/ernie-clm-base/ft_ernie-clm-base_${TASK}.sh > $LOG_DIR/ft_ernie-clm-base_${TASK}_${CKPT_NAME}.log 2>&1

echo "running QNLI..."
TASK="qnli"
bash run/ernie-clm-base/ft_ernie-clm-base_${TASK}.sh > $LOG_DIR/ft_ernie-clm-base_${TASK}_${CKPT_NAME}.log 2>&1

echo "running SST-2..."
TASK="sst2"
bash run/ernie-clm-base/ft_ernie-clm-base_${TASK}.sh > $LOG_DIR/ft_ernie-clm-base_${TASK}_${CKPT_NAME}.log 2>&1

echo "running CoLA..."
TASK="cola"
bash run/ernie-clm-base/ft_ernie-clm-base_${TASK}.sh > $LOG_DIR/ft_ernie-clm-base_${TASK}_${CKPT_NAME}.log 2>&1

echo "running STS-B..."
TASK="stsb"
bash run/ernie-clm-base/ft_ernie-clm-base_${TASK}.sh > $LOG_DIR/ft_ernie-clm-base_${TASK}_${CKPT_NAME}.log 2>&1

echo "running MRPC..."
TASK="mrpc"
bash run/ernie-clm-base/ft_ernie-clm-base_${TASK}.sh > $LOG_DIR/ft_ernie-clm-base_${TASK}_${CKPT_NAME}.log 2>&1

echo "running RTE..."
TASK="rte"
bash run/ernie-clm-base/ft_ernie-clm-base_${TASK}.sh > $LOG_DIR/ft_ernie-clm-base_${TASK}_${CKPT_NAME}.log 2>&1

echo "running WNLI..."
TASK="wnli"
bash run/ernie-clm-base/ft_ernie-clm-base_${TASK}.sh > $LOG_DIR/ft_ernie-clm-base_${TASK}_${CKPT_NAME}.log 2>&1

