#!/bin/bash

set -e

export PYTHONPATH=$PYTHONPATH:src/

CKPT_NAME=ckpt-25000
LOG_DIR=log/ft_ernie-pixel-only/$CKPT_NAME

mkdir -p $LOG_DIR

for TASK in mnli qqp qnli sst2 cola mrpc stsb rte wnli
do
    echo "running ${TASK}..."
    bash run/ernie-pixel-only/ft_ernie-pixel-only_${TASK}.sh > $LOG_DIR/ft_ernie-pixel-only_${TASK}_${CKPT_NAME}.log 2>&1
done

echo "finished!"