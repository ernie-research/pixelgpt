#!/bin/bash

set -e

pip install -r requirements.txt


# 安装 transformers==4.36.2
pip install transformers==4.36.2 datasets==2.16.0

# 安装 torch=1.13.1 和 torchvision
pip install torch==1.13.1 torchvision

# 安装 evluate
export https_proxy=http://172.19.57.45:3128;export http_proxy=http://172.19.57.45:3128
git clone https://github.com/huggingface/evaluate.git
unset https_proxy;unset http_proxy
cd evaluate && pip install -e .

# 安装tensorboard
pip install tensorboard