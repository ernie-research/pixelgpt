#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Conditional patch generation with the auto-regressive models of the library
"""


import argparse
import inspect
import logging
from typing import Tuple

import torch
from accelerate import PartialState
from accelerate.utils import set_seed

from base64 import b64decode
from PIL import Image
import io

from transformers import (
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    CTRLLMHeadModel,
    CTRLTokenizer,
    GenerationMixin,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTJForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    OPTForCausalLM,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.ernie_pixel import (
    ErniePixelConfig,
    ErniePixelForCausalLM,
)

from models.ernie_pixel.utils import (
    get_transforms,
    get_attention_mask,
    clip,
    patchify,
    unpatchify,
)
from models.ernie_pixel.rendering import (
    PyGameTextRenderer
)

from data_collator import (
    DataCollatorForPixelOnly,
    DataCollatorForErniePixel,
    # collate_fn
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "ernie-pixel": (ErniePixelForCausalLM, LlamaTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


#
# Functions to prepare models' input
#


PREPROCESSING_FUNCTIONS = {
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def sparse_model_config(model_config):
    embedding_size = None
    if hasattr(model_config, "hidden_size"):
        embedding_size = model_config.hidden_size
    elif hasattr(model_config, "n_embed"):
        embedding_size = model_config.n_embed
    elif hasattr(model_config, "n_embd"):
        embedding_size = model_config.n_embd

    num_head = None
    if hasattr(model_config, "num_attention_heads"):
        num_head = model_config.num_attention_heads
    elif hasattr(model_config, "n_head"):
        num_head = model_config.n_head

    if embedding_size is None or num_head is None or num_head == 0:
        raise ValueError("Check the model config")

    num_embedding_size_per_head = int(embedding_size / num_head)
    if hasattr(model_config, "n_layer"):
        num_layer = model_config.n_layer
    elif hasattr(model_config, "num_hidden_layers"):
        num_layer = model_config.num_hidden_layers
    else:
        raise ValueError("Number of hidden layers couldn't be determined from the model config")

    return num_layer, num_head, num_embedding_size_per_head


def generate_past_key_values(model, batch_size, seq_len):
    num_block_layers, num_attention_heads, num_embedding_size_per_head = sparse_model_config(model.config)
    if model.config.model_type == "bloom":
        past_key_values = tuple(
            (
                torch.empty(int(num_attention_heads * batch_size), num_embedding_size_per_head, seq_len)
                .to(model.dtype)
                .to(model.device),
                torch.empty(int(num_attention_heads * batch_size), seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    else:
        past_key_values = tuple(
            (
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    return past_key_values


def prepare_jit_inputs(inputs, model, tokenizer):
    batch_size = len(inputs)
    dummy_input = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
    dummy_input = dummy_input.to(model.device)
    if model.config.use_cache:
        dummy_input["past_key_values"] = generate_past_key_values(model, batch_size, 1)
    dummy_input["attention_mask"] = torch.cat(
        [
            torch.zeros(dummy_input["attention_mask"].shape[0], 1)
            .to(dummy_input["attention_mask"].dtype)
            .to(model.device),
            dummy_input["attention_mask"],
        ],
        -1,
    )
    return dummy_input


class _ModelFallbackWrapper(GenerationMixin):
    __slots__ = ("_optimized", "_default")

    def __init__(self, optimized, default):
        self._optimized = optimized
        self._default = default

    def __call__(self, *args, **kwargs):
        if kwargs["past_key_values"] is None and self._default.config.use_cache:
            kwargs["past_key_values"] = generate_past_key_values(self._default, kwargs["input_ids"].shape[0], 0)
        kwargs.pop("position_ids", None)
        for k in list(kwargs.keys()):
            if kwargs[k] is None or isinstance(kwargs[k], bool):
                kwargs.pop(k)
        outputs = self._optimized(**kwargs)
        lm_logits = outputs[0]
        past_key_values = outputs[1]
        fixed_output = CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
        return fixed_output

    def __getattr__(self, item):
        return getattr(self._default, item)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs
    ):
        return self._default.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self._default._reorder_cache(past_key_values, beam_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="ernie-pixel",
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument("--renderer_path", type=str, default="renderers/noto_renderer")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Whether or not to use cpu. If set to False, " "we will use gpu/npu or mps device if available",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--jit", action="store_true", help="Whether or not to use jit trace to accelerate inference")
    args = parser.parse_args()

    # Initialize the distributed state.
    distributed_state = PartialState(cpu=args.use_cpu)

    logger.warning(f"device: {distributed_state.device}, 16-bits inference: {args.fp16}")

    if args.seed is not None:
        set_seed(args.seed)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(args.model_name_or_path)

    # Set the model to the right device
    model.to(distributed_state.device)

    if args.fp16:
        model.half()
    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.length = adjust_length_to_model(args.length, max_sequence_length=max_seq_length)
    logger.info(args)

    args.prompt = """Baidu, Inc. (/ˈbaɪduː/ BY-doo; Chinese: 百 度; pinyin: Bǎidù, meaning "hundred times") is a Chinese multinational technology company specializing in Internet-related services, products, and artificial intelligence (AI), headquartered in Beijing's Haidian District.[3] It is one of the largest AI and Internet companies in the world. The holding company of the group is incorporated in the Cayman Islands.[2] Baidu was incorporated in January 2000 by Robin Li and Eric Xu. Baidu has origins in RankDex, an earlier search engine developed by Robin Li in 1996, before he founded Baidu in 2000.[4]

Baidu offers various services, including a Chinese search engine, as well as a mapping service called Baidu Maps. Baidu offers about 57 search and community services, such as Baidu Baike (an online encyclopedia), Baidu Wangpan (a cloud storage service), and Baidu Tieba (a keyword-based discussion forum).[5]

Baidu Global Business Unit (GBU) is responsible for Baidu's international products and services for markets outside of China. Baidu GBU's product portfolio includes keyboard apps Simeji and Facemoji Keyboard, content recommendation platform popIn, augmented reality network OmniAR, Japanese smart projector popIn Aladdin, and ad platform MediaGo, which is focused on Chinese advertisers looking to reach overseas users. In 2017, Baidu GBU entered into a partnership with Snap Inc. to act as the company's official ad reseller for Snapchat in Greater China, South Korea, Japan and Singapore.[6] The partnership was extended in 2019.[7]

In 2018, Baidu divested the "Global DU business" portion of its overseas business, which developed a series of utility apps including ES File Explorer, DU Caller, Mobojoy, Photo Wonder and DU Recorder, etc.[8] This business now operates independently of Baidu under the name DO Global.[9]

In December 2007, Baidu became the first Chinese company to be included in the NASDAQ-100 index.[10] As of May 2018, Baidu's market cap rose to US$99 billion.[11][12][13] In October 2018, Baidu became the first Chinese firm to join the United States-based computer ethics consortium Partnership on AI.[14]
] In 2001, Baidu allowed advertisers to bid for ad space and then pay Baidu every time a customer clicked on an ad, predating Google's approach to advertising.[20] In 2003, Baidu launched a news search engine and picture search engine, adopting a special identification technology capable of identifying and grouping the articles.[23]
different languages. Smaller than a typical smartphone, the 140-gram translation device can also be used as a portable Wi-Fi router and is able to operate on networks in 80 countries. It is still under development. Baidu will also be inserting artificial intelligence (AI) technology into smartphones, through its deep learning platform.[36][37] At the same period, it has also led a joint investment of US$12 billion with Alibaba Group, Tencent, JD.com and Didi Chuxing, acquiring 35% of China Unicom's stakes.[38][39][40]"""
#     args.prompt = """水调歌头·明月几时有
# 【作者】苏轼 【朝代】宋译文对照
# 丙辰中秋，欢饮达旦，大醉，作此篇，兼怀子由。

# 明月几时有？把酒问青天。不知天上宫阙，今夕是何年。我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。

# 转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。"""
#     args.prompt = """The OpenAI API is powered by GPT-3 language models which can be coaxed to perform natural language tasks using carefully engineered text prompts. But these models can also generate outputs that are untruthful, toxic, or reflect harmful sentiments. This is in part because GPT-3 is trained to predict the next word on a large dataset of Internet text, rather than to safely perform the language task that the user wants. In other words, these models aren’t aligned with their users.

# To make our models safer, more helpful, and more aligned, we use an existing technique called reinforcement learning from human feedback (RLHF). On prompts submitted by our customers to the API,A
# [A]
# We only use prompts submitted through the Playground to an earlier version of the InstructGPT models that was deployed in January 2021. Our human annotators remove personal identifiable information from all prompts before adding it to the training set.

#  our labelers provide demonstrations of the desired model behavior, and rank several outputs from our models. We then use this data to fine-tune GPT-3.

# The resulting InstructGPT models are much better at following instructions than GPT-3. They also make up facts less often, and show small decreases in toxic output generation. Our labelers prefer outputs from our 1.3B InstructGPT model over outputs from a 175B GPT-3 model, despite having more than 100x fewer parameters. At the same time, we show that we don’t have to compromise on GPT-3’s capabilities, as measured by our model’s performance on academic NLP evaluations.

# These InstructGPT models, which have been in beta on the API for more than a year, are now the default language models accessible on our API.B
# [B]
# The InstructGPT models deployed in the API are updated versions trained using the same human feedback data. They use a similar but slightly different training method that we will describe in a forthcoming publication.

#  We believe that fine-tuning language models with humans in the loop is a powerful tool for improving their safety and reliability, and we will continue to push in this direction.

# This is the first time our alignment research, which we’ve been pursuing for several years,1,2,3 has been applied to our product. Our work is also related to recent research that fine-tunes language models to follow instructions using academic NLP datasets, notably FLAN4 and T0.5 A key motivation for our work is to increase helpfulness and truthfulness while mitigating the harms and biases of language models.6,7,8,9,10 Some of our previous research in this direction found that we can reduce harmful outputs by fine-tuning on a small curated dataset of human demonstrations.11 Other research has focused on filtering the pre-training dataset,12 safety-specific control tokens,13,14 or steering model generations.15,16 We are exploring these ideas and others in our ongoing alignment research.

# Results
# We first evaluate how well outputs from InstructGPT follow user instructions, by having labelers compare its outputs to those from GPT-3. We find that InstructGPT models are significantly preferred on prompts submitted to both the InstructGPT and GPT-3 models on the API. This holds true when we add a prefix to the GPT-3 prompt so that it enters an “instruction-following mode.”"""
    
    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    # Load text renderer
    text_renderer = PyGameTextRenderer.from_pretrained(args.renderer_path)
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
    prefix = args.prefix if args.prefix else args.padding_text
    prompt_text = prefix + prompt_text
    encoding = text_renderer(text=prompt_text)
    pixel_values = [Image.fromarray(encoding.pixel_values)]
    print("image size:", pixel_values[0].size)
    # pixel_values[0].save("tmp/before.png")
    pixel_values = [transforms(x) for x in pixel_values]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(distributed_state.device)

    # from torchvision import transforms as T
    # to_pil=T.ToPILImage()
    # # to_pil(pixel_values.squeeze()).save("tmp/after.png")

    # labels = depatchify(patchify(pixel_values), image_height).squeeze().detach().cpu().squeeze()
    # print(labels)
    # label_img = to_pil(labels).save("tmp/label.png")
    # img = clip(prediction).numpy().astype('uint8')
    # img = Image.fromarray(img)
    
    ########## process text ###############
    # encoded_prompt = tokenizer.encode(, add_special_tokens=False, return_tensors="pt")
    # encoded_prompt = encoded_prompt.to(distributed_state.device)
    # if encoded_prompt.size()[-1] == 0:
    #     input_ids = None
    # else:
    #     input_ids = encoded_prompt
    ######################################

    if args.jit:
        jit_input_texts = ["enable jit"]
        jit_inputs = prepare_jit_inputs(jit_input_texts, model, tokenizer)
        torch._C._jit_set_texpr_fuser_enabled(False)
        model.config.return_dict = False
        if hasattr(model, "forward"):
            sig = inspect.signature(model.forward)
        else:
            sig = inspect.signature(model.__call__)
        jit_inputs = tuple(jit_inputs[key] for key in sig.parameters if jit_inputs.get(key, None) is not None)
        traced_model = torch.jit.trace(model, jit_inputs, strict=False)
        traced_model = torch.jit.freeze(traced_model.eval())
        traced_model(*jit_inputs)
        traced_model(*jit_inputs)

        model = _ModelFallbackWrapper(traced_model, model)

    output = model.generate_pixel(
        pixel_values=pixel_values,
        max_length=args.length + len(pixel_values[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )
    from torchvision import transforms as T
    to_pil=T.ToPILImage()
    resize = T.Resize(size=(image_height, image_width))
    prediction = clip(unpatchify(output["logits_pixel"], image_height).detach().cpu().squeeze())
    labels = unpatchify(patchify(pixel_values), image_height).detach().cpu().squeeze()
    print(labels.shape)
    to_pil(prediction).save("tmp/case1/pred1.png")
    to_pil(labels).save("tmp/case1/label1.png")
    # img = clip(prediction).numpy().astype('uint8')
    # img = Image.fromarray(img)
    return 

def depatchify(x: torch.Tensor, patch_size: int = 16):
    """
    x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
    or
    x: (L, patch_size**2 *3) imgs: (3, H, W)
    """
    is_single_image = len(x.shape) == 2
    if is_single_image:
        x = x.unsqueeze(0)

    h = p = patch_size
    # h = w = int(x.shape[1] ** 0.5)
    # assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], x.shape[1], p, p, 3))
    x = torch.einsum("hwpqc->chpwq", x)
    c, h, p, w, q = x.shape
    imgs = x.reshape(shape=(c, h*p, w*q))

    if is_single_image:
        return imgs.squeeze(0)
    return imgs


if __name__ == "__main__":
    main()
