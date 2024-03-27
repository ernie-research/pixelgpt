#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""

import argparse
from base64 import b64decode
import copy
from io import BytesIO
import io
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import datasets
import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset, load_metric, DatasetDict
from PIL import Image
from pixel import (
    # AutoConfig,
    # AutoModelForSequenceClassification,
    Modality,
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PIXELTrainer,
    PIXELTrainingArguments,
    PoolingMode,
    PyGameTextRenderer,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    log_sequence_classification_predictions,
    resize_model_embeddings,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    set_seed,
)

from evaluate import load
# ernie-pixel
from ernie_pixel import (
    ErniePixelConfig,
    ErniePixelForCausalLM,
    ErniePixelForSequenceClassification
)
# from ernie_pixel.rendering import (
#     PyGameTextRenderer,
# )

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


XNLI_LANGUAGES = ["en", "fr", "es", "el", "de", "bg", "ru", "tr", "ar", "vi", "th", "zh", "hi", "sw", "ur"]


task_to_keys = {
    "xnli": ("premise", "hypothesis"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=196,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    data_file_dir: Optional[str] = field(
        default=None, metadata={"help": "directory containing the data files"}
    )
    load_from_file: bool = field(
        default=False, metadata={"help": "Load dataset from file or not."}
    )
    render_mode: str = field(
        default="rgb", metadata={"help": "Render mode for the dataset. Options are 'rgb' or 'gray' or 'binary'."}
    )



    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as model_name"}
    )
    rendering_backend: Optional[str] = field(
        default="pangocairo", metadata={
            "help": "Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications it is "
                    "recommended to use the default 'pangocairo'."}
    )
    fallback_fonts_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing fallback font files used by the text renderer for PIXEL. "
                          "PyGame does not support fallback fonts so this argument is ignored when using the "
                          "PyGame backend."},
    )
    render_rgb: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to render images in RGB. RGB rendering can be useful when working with emoji "
            "but it makes rendering a bit slower, so it is recommended to turn on RGB rendering only "
            "when there is need for it. PyGame does not support fallback fonts so this argument is ignored "
            "when using the PyGame backend."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    pooling_mode: str = field(
        default="mean",
        metadata={
            "help": f"Pooling mode to use in classification head (options are {[e.value for e in PoolingMode]}."
        },
    )
    pooler_add_layer_norm: bool = field(
        default=True,
        metadata={
            "help": "Whether to add layer normalization to the classification head pooler. Note that this flag is"
            "ignored and no layer norm is added when using CLS pooling mode."
        },
    )
    dropout_prob: float = field(
        default=None, metadata={"help": "Dropout probability for attention blocks and classification head"}
    )
    model_type: str = field(
        default=None, metadata={"help": "Model type to use for the model. If not specified, it will be inferred from"}
    )
    modality: str = field(
        default=None, metadata={"help": "Modality to use for the model. If not specified, it will be inferred from"}
    )
    patch_size: int = field(
        default=None, metadata={"help": "Patch size to use for the model. If not specified, it will be set from model config"}
    )

    def __post_init__(self):
        self.pooling_mode = PoolingMode.from_string(self.pooling_mode)

        if not self.rendering_backend.lower() in ["pygame", "pangocairo"]:
            raise ValueError("Invalid rendering backend. Supported backends are 'pygame' and 'pangocairo'.")
        else:
            self.rendering_backend = self.rendering_backend.lower()


def get_processor(model_args: argparse.Namespace, modality: Modality):
    if modality == Modality.TEXT:
        processor = AutoTokenizer.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            use_fast=True,
            add_prefix_space=True if model_args.model_name_or_path == "roberta-base" else False,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
        )
        # 增加llama模型
        processor.pad_token = processor.eos_token
        logger.info("Set pad token to eos_token: {} ({})".format(processor.eos_token, processor.eos_token_id))
        # processor.padding_side = "left" # 
    elif modality == Modality.IMAGE:
        renderer_cls = PyGameTextRenderer if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
        processor = renderer_cls.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
            fallback_fonts_dir=model_args.fallback_fonts_dir,
            rgb=model_args.render_rgb,
        )
    else:
        raise ValueError("Modality not supported.")
    return processor


def get_model_and_config(model_args: argparse.Namespace, num_labels: int, task_name: str):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token if model_args.use_auth_token else None,
    }

    if model_args.model_type is None:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels, # ???
            finetuning_task=task_name, # ???
            **config_kwargs,
        )
    elif model_args.model_type == "ernie-pixel":
        config = ErniePixelConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels, # ???
            finetuning_task=task_name, # ???
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    if config.model_type in ["llama"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    elif config.model_type in ["ernie-pixel"]:
        if model_args.dropout_prob is not None:
            config.attention_dropout = model_args.dropout_prob
        if model_args.patch_size is not None:
            config.patch_size = model_args.patch_size
        model = ErniePixelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    return model, config



def get_collator(
    training_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
    is_regression: bool = False
):
    def image_collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        if "label" in examples[0]:
            if is_regression:
                labels = torch.FloatTensor([example["label"] for example in examples])
            else:
                labels = torch.LongTensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "attention_mask": attention_mask, "labels": labels}
        return {"pixel_values": pixel_values, "attention_mask": attention_mask}

    if modality == Modality.IMAGE:
        collator = image_collate_fn
    elif modality == Modality.TEXT:
        collator = DataCollatorWithPadding(processor, pad_to_multiple_of=8) if training_args.fp16 else None
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return collator
        
def image_preprocess_fn(
        example,
        data_args: argparse.Namespace,
        processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
        sentence_keys: Tuple[str, Optional[str]],
        ):
    sentence1_key, sentence2_key = sentence_keys
    transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )
    format_fn = glue_strip_spaces
    if sentence2_key:
        # encodings = [
        #     processor(text=(format_fn(a), format_fn(b)))
        #     for a, b in zip(examples[sentence1_key], examples[sentence2_key])
        # ]
        encoding = processor(text=(format_fn(example[sentence1_key]), format_fn(example[sentence2_key])))
    else:
        # encodings = [processor(text=format_fn(a)) for a in examples[sentence1_key]]
        encoding = processor(text=(format_fn(example[sentence1_key])))

    # examples["pixel_values"] = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
    example["pixel_values"] = transforms(Image.fromarray(encoding.pixel_values))
    
    example["attention_mask"] = get_attention_mask(encoding.num_text_patches, seq_length=data_args.max_seq_length)

    return example

def get_preprocess_fn(
    data_args: argparse.Namespace,
    processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizerFast],
    modality: Modality,
    sentence_keys: Tuple[str, Optional[str]],
):
    sentence1_key, sentence2_key = sentence_keys

    if modality == Modality.IMAGE:
        transforms = get_transforms(
            do_resize=True,
            size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
        )

        
        def image_preprocess_fn(examples):
            if data_args.render_mode == 'rgb':
                examples["pixel_values"] = [Image.open(io.BytesIO(b64decode(image))).convert('RGB') for image in examples["image"]]
            elif data_args.render_mode == 'gray':
                examples["pixel_values"] = [Image.open(io.BytesIO(b64decode(image))).convert('L') for image in examples["image"]]
            elif data_args.render_mode == 'binary':
                threshold_value = 127
                examples["pixel_values"] = [Image.open(io.BytesIO(b64decode(image))).convert('L').point(lambda x: 255 if x > threshold_value else 0, '1') for image in examples["image"]]
            
            
            examples["pixel_values"] = [transforms(image) for image in examples["pixel_values"]]
            examples["attention_mask"] = [get_attention_mask(num_patches, seq_length=data_args.max_seq_length) for num_patches in examples["num_patches"]]

            examples.pop("image")
            return examples

        preprocess_fn = image_preprocess_fn

    elif modality == Modality.TEXT:

        def text_preprocess_fn(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = processor(*args, padding=True, max_length=data_args.max_seq_length, truncation=True)

            if "label" in examples:
                result["label"] = [l for l in examples["label"]]


            return result

        preprocess_fn = text_preprocess_fn
    else:
        raise ValueError(f"Modality {modality} not supported.")

    return preprocess_fn


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PIXELTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # ================== Load Dataset ==================
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None and not data_args.load_from_file:
        # Downloading and loading a dataset from the hub.

        train_datasets = [
            load_dataset(
                '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/datasets--xnli/',
                lang,
                split="train",
                use_auth_token=model_args.use_auth_token,
            )
            for lang in XNLI_LANGUAGES
        ]
        eval_datasets = [
            load_dataset(
                '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/datasets--xnli/',
                lang,
                split="validation",
                use_auth_token=model_args.use_auth_token,
            )
            for lang in XNLI_LANGUAGES
        ]
        predict_datasets = [
            load_dataset(
                '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/datasets--xnli/',
                lang,
                split="test",
                use_auth_token=model_args.use_auth_token,
            )
            for lang in XNLI_LANGUAGES
        ]
        
        raw_datasets = DatasetDict({
            "train": concatenate_datasets(train_datasets),
            "validation": concatenate_datasets(eval_datasets),
            "test": concatenate_datasets(predict_datasets),
        })
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        # Loading a dataset from local json files
        train_datasets = [
            load_dataset(
                "json",
                data_files=os.path.join(data_args.data_file_dir,  'xnli' + '-' + lang + "-" + "train", "part-00000.gz")
            )["train"]
            for lang in XNLI_LANGUAGES
        ]

        eval_datasets = [
            load_dataset(
                "json",
                data_files=os.path.join(data_args.data_file_dir,  'xnli' + '-' + lang + "-" + "validation", "part-00000.gz")
            )["train"]
            for lang in XNLI_LANGUAGES
        ]

        predict_datasets = [
            load_dataset(
                "json",
                data_files=os.path.join(data_args.data_file_dir,  'xnli' + '-' + lang + "-" + "test", "part-00000.gz")
            )["train"]
            for lang in XNLI_LANGUAGES
        ]

        raw_datasets = DatasetDict({
            "train": concatenate_datasets(train_datasets),
            "validation": concatenate_datasets(eval_datasets),
            "test": concatenate_datasets(predict_datasets),
        })

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.



    # Labels
    if data_args.task_name is not None and not data_args.load_from_file:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets['train'].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets['train'].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and config
    model, config = get_model_and_config(model_args, num_labels, data_args.task_name)


    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    # ======== 设置 model.pad_token_id ===================================================
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    # ===================================================================================

    # ======= 设置 modality ==============================================================
    # modality = Modality.TEXT if config.model_type in ["bert", "roberta"] else Modality.IMAGE
    if model_args.modality is None:
        modality = Modality.TEXT if config.model_type in ["bert", "roberta", "llama"] else Modality.IMAGE #增加gpt2
    else:
        if model_args.modality == "text":
            modality = Modality.TEXT
        elif model_args.modality == "image":
            modality = Modality.IMAGE
        else:
            raise ValueError(f"modality {model_args.modality} not supported")
        
    # ========== 如果模型为`pixel-based`且输入为单通道，则按通道平均 patch embedding 的权重 ==========
    if modality in [Modality.IMAGE] and data_args.render_mode in ['gray', 'binary']:
        """Average the projection weight for channels=3 into a single channel projection."""
        from torch import nn
      
        # 获取卷积核权重，其形状为 [out_channels, in_channels, height, width]
        projection = model.model.embed_patches.patch_embeddings.projection
        original_weights = projection.weight.data

        # 计算新的单通道卷积核权重, 对每个输出通道的卷积核，将其三个输入通道的参数求平均
        new_weights = original_weights.mean(dim=1, keepdim=True)  # 沿着输入通道维度进行平均

        # 创建新的卷积层，其输入通道数为1，将计算出的平均新权重赋值给新卷积层
        out_channels, kernel_size, stride = projection.out_channels, projection.kernel_size, projection.stride
        new_conv = nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride)
        new_conv.weight.data = new_weights

        # 如果有偏置项，也要复制过来
        if projection.bias is not None:
            new_conv.bias.data = projection.bias.data
        
        model.model.embed_patches.patch_embeddings.projection = new_conv
        
        # 更新`num_channels`参数
        model.model.embed_patches.patch_embeddings.num_channels = 1
        model.config.num_channels = 1

    # ========== 打印模型参数量 ==============
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(model))
        
    processor = get_processor(model_args, modality)

    if modality == Modality.IMAGE:
        if processor.max_seq_length != data_args.max_seq_length:
            processor.max_seq_length = data_args.max_seq_length

        # resize_model_embeddings(model, processor.max_seq_length)
    # ===================================================================================

    # ======== Preprocessing the datasets ========
    preprocess_fn = get_preprocess_fn(data_args, processor, modality, (sentence1_key, sentence2_key))
    
    if training_args.do_train:
        train_dataset = raw_datasets['train']
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        if modality == Modality.IMAGE:
            train_dataset.features["pixel_values"] = datasets.Image()
        train_dataset.set_transform(preprocess_fn)

    if training_args.do_eval:
        eval_dataset = raw_datasets['validation']
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
            eval_datasets = [eval_dataset.select(range(data_args.max_eval_samples)) for eval_dataset in eval_datasets ]
        if modality == Modality.IMAGE:
            eval_dataset.features["pixel_values"] = datasets.Image()
        eval_examples_l = [copy.deepcopy(e) for e in eval_datasets]
        eval_dataset.set_transform(preprocess_fn)
        [e.set_transform(preprocess_fn) for e in eval_datasets]

    if training_args.do_predict:
        predict_dataset = raw_datasets['test']
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
            predict_datasets = [predict_dataset.select(range(data_args.max_predict_samples)) for predict_dataset in predict_datasets ]
        if modality == Modality.IMAGE:
            predict_dataset.features["pixel_values"] = datasets.Image()
        predict_examples_l = [copy.deepcopy(e) for e in predict_datasets]
        predict_dataset.set_transform(preprocess_fn)
        [e.set_transform(preprocess_fn) for e in predict_datasets]

    # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # if training_args.do_eval:
    #     for index in random.sample(range(len(eval_dataset)), 3):
    #         logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")

    # Get the metric function
    metric = load("/root/paddlejob/workspace/liuqingyi01/code/ernie-pixel-ft/evaluate/metrics/xnli/xnli.py")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = PIXELTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        data_collator=get_collator(training_args, processor, modality, is_regression=is_regression),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
        if training_args.early_stopping
        else None,
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        for lang, eval_dataset, eval_examples in zip(XNLI_LANGUAGES, eval_datasets, eval_examples_l):
            logger.info(f"Evaluating {lang}")

            outputs = trainer.predict(test_dataset=eval_dataset, metric_key_prefix=f"eval_{lang}")
            metrics = outputs.metrics

            # Log predictions to understand where model goes wrong
            if training_args.log_predictions:
                log_sequence_classification_predictions(
                    training_args=training_args,
                    features=eval_dataset,
                    examples=eval_examples,
                    predictions=outputs.predictions,
                    sentence1_key=sentence1_key,
                    sentence2_key=sentence2_key,
                    modality=modality,
                    prefix=f"eval_{lang}",
                )

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics(f"eval_{lang}", metrics)
            trainer.save_metrics(f"eval_{lang}", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        for lang, predict_dataset, predict_examples in zip(XNLI_LANGUAGES, predict_datasets, predict_examples_l):
            logger.info(f"Predicting {lang}")

            outputs = trainer.predict(test_dataset=predict_dataset, metric_key_prefix=f"test_{lang}")
            metrics = outputs.metrics

            # Log predictions to understand where model goes wrong
            if training_args.log_predictions:
                log_sequence_classification_predictions(
                    training_args=training_args,
                    features=predict_dataset,
                    examples=predict_examples,
                    predictions=outputs.predictions,
                    sentence1_key=sentence1_key,
                    sentence2_key=sentence2_key,
                    modality=modality,
                    prefix=f"test_{lang}",
                )

            max_predict_samples = (
                data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(eval_dataset))

            trainer.log_metrics(f"test_{lang}", metrics)
            trainer.save_metrics(f"test_{lang}", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.dataset_name is not None:
        kwargs["language"] = XNLI_LANGUAGES
        kwargs["dataset_tags"] = "xnli-translate-train-all"
        kwargs["dataset_args"] = "xnli-translate-train-all"
        kwargs["dataset"] = "xnli-translate-train-all"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
