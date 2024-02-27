#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  //    ~    /  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
@author: Yekun Chai
@email: chaiyekun@baidu.com
@license: (C)Copyright Baidu NLP
@file: run_pixel_generation.py
@time: 2024/01/03 23:07:39
@desc: ernie-pixel pixel generation
'''

from collections import defaultdict
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict
from base64 import b64decode
from PIL import Image
import io

import sklearn

import datasets
import evaluate
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    # Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)

from models.ernie_pixel import (
    ErniePixelConfig,
    ErniePixelForCausalLM,
)

from data_collator import (
    DataCollatorForPixelOnly,
    DataCollatorForErniePixel,
    # collate_fn
)
from models.ernie_pixel import (
    ErniePixelConfig,
    ErniePixelForCausalLM,
)

from models.ernie_pixel.utils import (
    get_transforms,
    get_attention_mask,
)
from models.ernie_pixel.rendering import (
    PyGameTextRenderer
)

from utils import timeit

import logging
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


set_seed(42)


################
# params
text_renderer_name_or_path="renderers/noto_renderer"

text = "The capital city of China is"
#############


# Load text renderer
text_renderer = PyGameTextRenderer.from_pretrained(text_renderer_name_or_path)
# Set transformations --- resize by default and optionally apply normalization



max_pixels = text_renderer.pixels_per_patch * text_renderer.max_seq_length - 2 * text_renderer.pixels_per_patch
target_seq_length = max_pixels

image_height = text_renderer.pixels_per_patch
image_width = text_renderer.pixels_per_patch * text_renderer.max_seq_length


transforms = get_transforms(
    do_resize=True,
    size=(image_height, image_width),
    do_normalize=False,
    image_mean=None,
    image_std=None,
)

# data = {"pixel_values": [], "num_patches": [], 'text': []}
batch = {}
encoding = text_renderer(text=text)
        
batch["pixel_values"] = [transforms(Image.fromarray(encoding.pixel_values))]
# batch["num_patches"].append(encoding.num_text_patches)
# collate
batch["pixel_values"] = torch.stack(batch["pixel_values"])

config = ErniePixelConfig.from_pretrained(model_name_or_path)
config.image_size = (image_height, image_width)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = ErniePixelForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
)

# todo: pixel.generate implementation
output = model(**batch)
pass