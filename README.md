# [EMNLP'24 | Autoregressive Pre-Training on Pixels and Texts](https://arxiv.org/abs/2404.10710)

   <a href="https://huggingface.co/baidu" target="_blank">
      <img alt="Models" src="https://img.shields.io/badge/🤗-Models-blue" />
   </a>    
   <a href="https://huggingface.co/datasets/baidu/rendered_GLUE" target="_blank">
      <img alt="Datasets" src="https://img.shields.io/badge/📚-Data(rendered GLUE)-green" />
   </a> 
   <a href="https://huggingface.co/datasets/baidu/rendered_xnli" target="_blank">
      <img alt="Datasets" src="https://img.shields.io/badge/📚-Data(rendered XNLI)-orange" />
   </a> 
   <a href="https://arxiv.org/abs/2404.10710" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/📜-Paper-purple" /></a>
  <a href="https://2024.emnlp.org/" target="_blank"> <img alt="EMNLP 2024" src="https://img.shields.io/badge/Proceedings-EMNLP2024-red" /> </a>


The official repository which contains the code and model checkpoints for our paper [Autoregressive Pre-Training on Pixels and Texts (EMNLP 2024)](https://arxiv.org/pdf/2404.10710).


## 🔥 News
* **21 September, 2024:** 🎉 Our work has been accepted to [EMNLP 2024](https://2024.emnlp.org/)! 🎉
* **1 May, 2024:** 🎉 We release the official codebase and model weights of [`PixelGPT`](https://huggingface.co/baidu/PixelGPT), [`MonoGPT`](https://huggingface.co/baidu/MonoGPT), and [`DualGPT`](https://huggingface.co/baidu/DualGPT) . Stay tuned!🔥
  
<img width="634" alt="image" src="https://github.com/ernie-research/pixelgpt/assets/13767887/1827f43f-34ca-448e-80ca-1fd0a523f213">


Harnessing visual texts represents a burgeoning frontier in the evolution of language modeling. In this paper, we introduce a novel pre-training framework for a suite of pixel-based autoregressive language models, pre-training on a corpus of over 400 million documents rendered as RGB images. Our approach is characterized by a dual-modality training regimen, engaging both visual data through next patch prediction with a regression head and textual data via next token prediction with a classification head. This study is particularly focused on investigating the synergistic interplay between visual and textual modalities of language. Our comprehensive evaluation across a diverse array of benchmarks reveals that the confluence of visual and textual data substantially augments the efficacy of pixel-based language models. Notably, our findings show that a unidirectional pixel-based model, _devoid_ of textual data during training, can match the performance levels of advanced bidirectional pixel-based models on various language understanding benchmarks. This work highlights the considerable untapped potential of integrating visual and textual information for language modeling purposes. We will release our code, data, and checkpoints to inspire further research advancement.

## 📕 Requirements
To set up the environment and install dependencies, run:
```
bash run_requirements.sh
```
## 📚 Fine-tuning Data
We fine-tune PixelGPT on the rendered GLUE and XNLI datasets. These rendered versions are publicly available at [baidu/rendered_GLUE](https://huggingface.co/datasets/baidu/rendered_GLUE) and [baidu/rendered_xnli](https://huggingface.co/datasets/baidu/rendered_xnli). After downloading the datasets from HuggingFace, extract them locally:
```
# Extract rendered GLUE
tar -xvf rendered_glue.tar

# Extract rendered XNLI
tar -xvf rendered_xnli.tar
```
For the rendered GLUE dataset, the extracted files contain multiple tasks. Each task has a corresponding training set, validation set, and test set. Note that for the MNLI task, both the validation and test sets contain matched and mismatched versions. You will need to assign the local paths of these task datasets to the `--train_file`, `--validation_file`, and `--test_file` parameters in the fine-tuning script.
For the rendered XNLI dataset, assign the local dataset path to the `--data_file_dir` parameter in the corresponding fine-tuning script.
## 📌 Pre-trained Models
We pre-trained PixelGPT and three other models: MonoGPT, and DualGPT. We release checkpoints used in our experiment, which can be downloaded at [baidu/PixelGPT](https://huggingface.co/baidu/PixelGPT), [baidu/MonoGPT](https://huggingface.co/baidu/MonoGPT), and [baidu/DualGPT](https://huggingface.co/baidu/DualGPT). Before running the fine-tuning scripts bellow, download the corresponding pre-trained models from our open-source model repository above and place the file in the pre-trained model directory, e.g. `pretrained_models/pixel_gpt`.

## 🚀 Fine-tuning
Our main fine-tuning experiments were performed on rendered GLUE and XNLI. The scripts to run the experiments are given below.
### GLUE 
For example, to fine-tune on the MNLI task:
#### PixelGPT
```
bash run/pixel_gpt/ft_pixel_gpt_mnli.sh pretrained_models/PixelGPT
```
#### MonoGPT
```
# Text-only Fine-tuning
run/mono_gpt/ft_mono_gpt_mnli_text.sh pretrained_models/MonoGPT

# Pixel-only Fine-tuning
run/mono_gpt/ft_mono_gpt_mnli_pixel.sh pretrained_models/MonoGPT

# Pair-modality Fine-tuning
run/mono_gpt/ft_mono_gpt_mnli_pair.sh pretrained_models/MonoGPT
```

#### DualGPT
```
# Text-only Fine-tuning
run/dual_gpt/ft_dual_gpt_mnli_text.sh pretrained_models/DualGPT

# Pixel-only Fine-tuning
run/dual_gpt/ft_dual_gpt_mnli_pixel.sh pretrained_models/DualGPT


# Pair-modality Fine-tuning
run/dual_gpt/ft_dual_gpt_mnli_pair.sh pretrained_models/DualGPT
```


### XNLI Training
We evaluated XNLI in two settings: (1) *Translate-train-all*, where the model is fine-tuned on a combination of English and machine-translated data from 14 other languages; (2) *Cross-lingual Transfer* settings, where the model is fine-tuned only on English data and tested on multiple languages.

#### 1. Translate-train-all
##### PixelGPT
```
bash run/cross_lingual/xnli/train_all/pixel_gpt/ft_pixel_gpt_xnli.sh pretrained_models/PixelGPT
```
##### MonoGPT
```
# Text-only Fine-tuning
bash run/cross_lingual/xnli/train_all/mono_gpt/ft_mono_gpt_xnli_text.sh pretrained_models/MonoGPT

# Pixel-only Fine-tuning
bash run/cross_lingual/xnli/train_all/mono_gpt/ft_mono_gpt_xnli_image.sh pretrained_models/MonoGPT

# Pair-modality Fine-tuning
bash run/cross_lingual/xnli/train_all/mono_gpt/ft_mono_gpt_xnli_pair.sh pretrained_models/MonoGPT
```
##### DualGPT
```
# Text-only Fine-tuning
bash run/cross_lingual/xnli/train_all/dual_gpt/ft_dual_gpt_xnli_text.sh pretrained_models/DualGPT

# Pixel-only Fine-tuning
bash run/cross_lingual/xnli/train_all/dual_gpt/ft_dual_gpt_xnli_image.sh pretrained_models/DualGPT

# Pair-modality Fine-tuning
bash run/cross_lingual/xnli/train_all/dual_gpt/ft_dual_gpt_xnli_pair.sh pretrained_models/DualGPT
```

#### 2. Cross-lingaul Transfer
##### PixelGPT
```
bash run/cross_lingual/xnli/train_en/pixel_gpt/ft_pixel_gpt_xnli.sh pretrained_models/PixelGPT
```
##### MonoGPT
```
# Text-only Fine-tuning
bash run/cross_lingual/xnli/train_en/mono_gpt/ft_mono_gpt_xnli_text.sh pretrained_models/MonoGPT

# Pixel-only Fine-tuning
bash run/cross_lingual/xnli/train_en/mono_gpt/ft_mono_gpt_xnli_image.sh pretrained_models/MonoGPT

# Pair-modality Fine-tuning
run/cross_lingual/xnli/train_en/mono_gpt/ft_mono_gpt_xnli_pair.sh pretrained_models/MonoGPT
```
##### DualGPT
```
# Text-only Fine-tuning
bash run/cross_lingual/xnli/train_en/dual_gpt/ft_dual_gpt_xnli_text.sh pretrained_models/DualGPT

# Pixel-only Fine-tuning
bash run/cross_lingual/xnli/train_en/dual_gpt/ft_dual_gpt_xnli_image.sh pretrained_models/DualGPT

# Pair-modality Fine-tuning
bash run/cross_lingual/xnli/train_en/dual_gpt/ft_dual_gpt_xnli_pair.sh pretrained_models/DualGPT
```
## Citation
For attribution in academic contexts, please cite this work as:
```
@misc{chai2024autoregressivepretrainingpixelstexts,
  title = {Autoregressive Pre-Training on Pixels and Texts},
  author = {Chai, Yekun and Liu, Qingyi and Xiao, Jingwu and Wang, Shuohuan and Sun, Yu and Wu, Hua},
  year = {2024},
  eprint = {2404.10710},
  archiveprefix = {arXiv},
  primaryclass = {cs.CL},
  url = {https://arxiv.org/abs/2404.10710},
}
```
