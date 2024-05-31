# pixelgpt
![image](https://github.com/ernie-research/pixelgpt/blob/main/src/PixelGPT.png)
Harnessing visual texts represents a burgeoning frontier in the evolution of language modeling. In this paper, we introduce a novel pre-training framework for a suite of pixel-based autoregressive language models, pre-training on a corpus of over 400 million documents rendered as RGB images. Our approach is characterized by a dual-modality training regimen, engaging both visual data through next patch prediction with a regression head and textual data via next token prediction with a classification head. This study is particularly focused on investigating the synergistic interplay between visual and textual modalities of language. Our comprehensive evaluation across a diverse array of benchmarks reveals that the confluence of visual and textual data substantially augments the efficacy of pixel-based language models. Notably, our findings show that a unidirectional pixel-based model, _devoid_ of textual data during training, can match the performance levels of advanced bidirectional pixel-based models on various language understanding benchmarks. This work highlights the considerable untapped potential of integrating visual and textual information for language modeling purposes. We will release our code, data, and checkpoints to inspire further research advancement.
# Requirements
To run the code, you should install the dependency libraries.
```
bash run_requirements.sh
```
# Fine-tuning Data
We mainly fine-tune pixelgpt on rendered GLEU and XNLI datasets. The rendered version of these experimental datasets is released at [baidu/rendered_GLUE](https://huggingface.co/datasets/baidu/rendered_GLUE) and [baidu/rendered_xnli](https://huggingface.co/datasets/baidu/rendered_xnli).
# Pre-trained Models
We pre-trained PixelGPT and three other models: MonoGPT, and DualGPT. We release checkpoints used in our experiment, which can be downloaded at [baidu/PixelGPT](https://huggingface.co/baidu/PixelGPT), [baidu/MonoGPT](https://huggingface.co/baidu/MonoGPT), and [baidu/DualGPT](https://huggingface.co/baidu/DualGPT).
# Fine-tuning
Our main fine-tuning experiments were performed on rendered GLUE and XNLI. The scripts to run the experiments are given below. Before running the scripts, download the corresponding pre-trained models from our open-source model repository above and place the file in the pre-trained model directory, e.g. `pretrained_models/pixel_gpt`.
## GLEU 
Unless otherwise specified, we take the MNLI dataset as an example.
### PixelGPT
```
bash run/pixel_gpt/ft_pixel_gpt_mnli.sh pretrained_models/PixelGPT
```
### MonoGPT
- text-only fine-tuning
```
run/mono_gpt/ft_mono_gpt_mnli_text.sh pretrained_models/MonoGPT
```
- pixel-only fine-tuning
```
run/mono_gpt/ft_mono_gpt_mnli_pixel.sh pretrained_models/MonoGPT
```
- pair-modality fine-tuning
```
run/mono_gpt/ft_mono_gpt_mnli_pair.sh pretrained_models/MonoGPT
```

### DualGPT
- text-only fine-tuning
```
run/dual_gpt/ft_dual_gpt_mnli_text.sh pretrained_models/DualGPT
```
- pixel-only fine-tuning
```
run/dual_gpt/ft_dual_gpt_mnli_pixel.sh pretrained_models/DualGPT
```
- pair-modality fine-tuning
```
run/dual_gpt/ft_dual_gpt_mnli_pair.sh pretrained_models/DualGPT
```


## XNLI
our evaluation of rendered XNLI is performed in two distinct scenarios: (1) _Translate-train-all_, where the model is fine-tuned on a blend of original English and machine-translated data from other 14 languages, aiming to appraise the model's multilingual understanding; (2) _Cross-lingual Transfer_ settings, wherein fine-tuning is conducted solely on English data, with multi-language test sets employed to evaluate the model’s transferability across languages.  
### Translate-train-all
#### PixelGPT
```
bash run/cross_lingual/xnli/train_all/pixel_gpt/ft_pixel_gpt_xnli.sh pretrained_models/PixelGPT
```
#### MonoGPT
- text-only fine-tuning
```
bash run/cross_lingual/xnli/train_all/mono_gpt/ft_mono_gpt_xnli_text.sh pretrained_models/MonoGPT
```
- pixel-only fine-tuning
```
bash run/cross_lingual/xnli/train_all/mono_gpt/ft_mono_gpt_xnli_image.sh pretrained_models/MonoGPT
```
- pair-modality fine-tuning
```
bash run/cross_lingual/xnli/train_all/mono_gpt/ft_mono_gpt_xnli_pair.sh pretrained_models/MonoGPT
```
#### DualGPT
- text-only fine-tuning
```
bash run/cross_lingual/xnli/train_all/dual_gpt/ft_dual_gpt_xnli_text.sh pretrained_models/DualGPT
```
- pixel-only fine-tuning
```
bash run/cross_lingual/xnli/train_all/dual_gpt/ft_dual_gpt_xnli_image.sh pretrained_models/DualGPT
```
- pair-modality fine-tuning
```
bash run/cross_lingual/xnli/train_all/dual_gpt/ft_dual_gpt_xnli_pair.sh pretrained_models/DualGPT
```

### Cross-lingaul Transfer
#### PixelGPT
```
bash run/cross_lingual/xnli/train_en/pixel_gpt/ft_pixel_gpt_xnli.sh pretrained_models/PixelGPT
```
#### MonoGPT
```
# Text-only Fine-tuning
bash run/cross_lingual/xnli/train_en/mono_gpt/ft_mono_gpt_xnli_text.sh pretrained_models/MonoGPT

# Pixel-only Fine-tuning
bash run/cross_lingual/xnli/train_en/mono_gpt/ft_mono_gpt_xnli_image.sh pretrained_models/MonoGPT

# Pair-modality Fine-tuning
run/cross_lingual/xnli/train_en/mono_gpt/ft_mono_gpt_xnli_pair.sh pretrained_models/MonoGPT
```
#### DualGPT
- text-only
```
bash run/cross_lingual/xnli/train_en/dual_gpt/ft_dual_gpt_xnli_text.sh pretrained_models/DualGPT
```
- pixel-only
```
bash run/cross_lingual/xnli/train_en/dual_gpt/ft_dual_gpt_xnli_image.sh pretrained_models/DualGPT
```
- pair
```
bash run/cross_lingual/xnli/train_en/dual_gpt/ft_dual_gpt_xnli_pair.sh pretrained_models/DualGPT
```
# Citation
For attribution in academic contexts, please cite this work as:
```
@article{chai2024dual,
  title={Dual Modalities of Text: Visual and Textual Generative Pre-training},
  author={Chai, Yekun and Liu, Qingyi and Xiao, Jingwu and Wang, Shuohuan and Sun, Yu and Wu, Hua},
  journal={arXiv preprint arXiv:2404.10710},
  year={2024}
}
```
