# pixelgpt
![image](https://github.com/ernie-research/pixelgpt/blob/main/src/PixelGPT.png)
Harnessing visual texts represents a burgeoning frontier in the evolution of language modeling. In this paper, we introduce a novel pre-training framework for a suite of pixel-based autoregressive language models, pre-training on a corpus of over 400 million documents rendered as RGB images. Our approach is characterized by a dual-modality training regimen, engaging both visual data through next patch prediction with a regression head and textual data via next token prediction with a classification head. This study is particularly focused on investigating the synergistic interplay between visual and textual modalities of language. Our comprehensive evaluation across a diverse array of benchmarks reveals that the confluence of visual and textual data substantially augments the efficacy of pixel-based language models. Notably, our findings show that a unidirectional pixel-based model, _devoid_ of textual data during training, can match the performance levels of advanced bidirectional pixel-based models on various language understanding benchmarks. This work highlights the considerable untapped potential of integrating visual and textual information for language modeling purposes. We will release our code, data, and checkpoints to inspire further research advancement.
# Requirements
To run the code, you should install the dependency libraries.
```
bash run_requirements.sh
```
# Fine-tuning Data
We fine-tune pixelgpt on GLEU and XNLI datasets. The rendered version of these experimental datasets is released at [huggingface](https://huggingface.co/datasets/baidu/PixelGPT_sft).
# Models
We pre-trained PixelGPT and three other models: TextGPT, MonoGPT, and DualGPT. We release checkpoints used in our experiment and they can be downloaded at [Model](https://huggingface.co/baidu/PixelGPT).
# Fine-tuning
Our main fine-tuning experiments were performed on GLUE and XNLI. The scripts to run the experiments are given below. Note that before running the scripts, you should download the corresponding pre-trained models from our open-source model repository and place the model file in the pre-trained model directory, e.g. `pretrained_models/pixel_gpt`.
## GLEU 
Unless otherwise specified, we take the MNLI dataset as an example.
### TextGPT
```
bash run/text_gpt/ft_text_gpt_mnli.sh 
```
### PixelGPT
```
bash run/pixel_gpt/ft_pixel_gpt_mnli.sh
```
### MonoGPT
- text only
```
run/mono_gpt/ft_mono_gpt_mnli_text.sh
```
- pixel only
```
run/mono_gpt/ft_mono_gpt_mnli_pixel.sh
```
- pair
```
run/mono_gpt/ft_mono_gpt_mnli_pair.sh
```

### DualGPT
- text only
```
run/dual_gpt/ft_dual_gpt_mnli_text.sh
```
- pixel only
```
run/dual_gpt/ft_dual_gpt_mnli_pixel.sh
```
- pair
```
run/dual_gpt/ft_dual_gpt_mnli_pair.sh
```


## XNLI
we fine-tuned pixelgpt on XNLI with two settings: `Translate-train-all` and `Cross-lingual transfer`.
### Translate-train-all
#### TextGPT
```
bash run/cross_lingual/xnli/train_all/text_gpt/ft_text_gpt_xnli.sh
```
#### PixelGPT
```
bash run/cross_lingual/xnli/train_all/pixel_gpt/ft_pixel_gpt_xnli.sh
```
#### MonoGPT
- text-only
```
bash run/cross_lingual/xnli/train_all/mono_gpt/ft_mono_gpt_xnli_text.sh
```
- pixel-only
```
bash run/cross_lingual/xnli/train_all/mono_gpt/ft_mono_gpt_xnli_image.sh
```
- pair
```
bash run/cross_lingual/xnli/train_all/mono_gpt/ft_mono_gpt_xnli_pair.sh
```
#### DualGPT
- text-only
```
bash run/cross_lingual/xnli/train_all/dual_gpt/ft_dual_gpt_xnli_text.sh
```
- pixel-only
```
bash run/cross_lingual/xnli/train_all/dual_gpt/ft_dual_gpt_xnli_image.sh
```
- pair
```
bash run/cross_lingual/xnli/train_all/dual_gpt/ft_dual_gpt_xnli_pair.sh
```

### Cross-lingaul Transfer
#### TextGPT
```
bash run/cross_lingual/xnli/train_en/text_gpt/ft_text_gpt_xnli.sh
```
#### PixelGPT
```
bash run/cross_lingual/xnli/train_en/pixel_gpt/ft_pixel_gpt_xnli.sh
```
#### MonoGPT
- text-only
```
bash run/cross_lingual/xnli/train_en/mono_gpt/ft_mono_gpt_xnli_text.sh
```
- pixel-only
```
bash run/cross_lingual/xnli/train_en/mono_gpt/ft_mono_gpt_xnli_image.sh
```
- pair
```
bash run/cross_lingual/xnli/train_en/mono_gpt/ft_mono_gpt_xnli_pair.sh
```
#### DualGPT
- text-only
```
bash run/cross_lingual/xnli/train_en/dual_gpt/ft_dual_gpt_xnli_text.sh
```
- pixel-only
```
bash run/cross_lingual/xnli/train_en/dual_gpt/ft_dual_gpt_xnli_image.sh
```
- pair
```
bash run/cross_lingual/xnli/train_en/dual_gpt/ft_dual_gpt_xnli_pair.sh
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
