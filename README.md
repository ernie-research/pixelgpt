# pixelgpt
![image](https://github.com/ernie-research/pixelgpt/blob/main/src/PixelGPT_02.png)
Harnessing visual texts represents a burgeoning frontier in the evolution of language modeling. In this paper, we introduce a novel pre-training framework for a suite of pixel-based autoregressive language models, pre-training on a corpus of over 400 million documents rendered as RGB images. Our approach is characterized by a dual-modality training regimen, engaging both visual data through next patch prediction with a regression head and textual data via next token prediction with a classification head. This study is particularly focused on investigating the synergistic interplay between visual and textual modalities of language. Our comprehensive evaluation across a diverse array of benchmarks reveals that the confluence of visual and textual data substantially augments the efficacy of pixel-based language models. Notably, our findings show that a unidirectional pixel-based model, \textit{devoid} of textual data during training, can match the performance levels of advanced bidirectional pixel-based models on various language understanding benchmarks. This work highlights the considerable untapped potential of integrating visual and textual information for language modeling purposes. We will release our code, data, and checkpoints to inspire further research advancement.
# Requirements
To run the code, you should install the dependency libraries.
```
bash run_requirements.sh
```
# Fine-tuning Data
We fine-tune pixelgpt on GLEU and XNLI datasets. The rendered version of these experimental datasets is released at [huggingface](https://huggingface.co/datasets/baidu/PixelGPT_sft).
# Fine-tuning
## GLEU 
We Take the MNLI dataset as an example.
### TextGPT
```
bash run/ernie-clm-base/ft_ernie-clm-base_mnli.sh 
    --NUM_NODE=8
    --MASTER_POART=23451
    --MODALITY="text"
    --TASK="mnli"
    --MODEL=$1 # also works with "bert-base-cased", "roberta-base", etc.
    --RENDERING_BACKEND="pygame"  # Consider trying out both "pygame" and "pangocairo" to see which one works best
    --SEQ_LEN=768
    --BSZ=8
    --GRAD_ACCUM=None  # We found that higher batch sizes can sometimes make training more stable
    --LR=None
    --SEED=42
    --MAX_STEPS=None
    --WARMUP_STEPS=100
    --EVAL_STEPS=500
    --SAVE_STEPS=500
    --IS_EARLY_STOPPING=True
    --METRIC_FOR_BEST_MODEL="eval_accuracy"
    --EARLY_STOPPING_PATIENCE=8
    --GREATER_IS_BETTER=True
```
### ernie-pixel-only
```
bash run/ernie-clm-base/ft_ernie-clm-base_mnli.sh
```
### ernie-pixel-mono
- text only
```
run/ernie-pixel-mono/ft_ernie-pixel-mono_mnli_text.sh
```
- pixel only
```
run/ernie-pixel-mono/ft_ernie-pixel-mono_mnli_pixel.sh
```
- pair
```
run/ernie-pixel-mono/ft_ernie-pixel-mono_mnli_pair.sh
```

### ernie-pixel-clm
- text only
```
run/ernie-pixel-clm/ft_ernie-pixel-clm_mnli_text.sh
```
- pixel only
```
run/ernie-pixel-clm/ft_ernie-pixel-clm_mnli_pixel.sh
```
- pair
```
run/ernie-pixel-clm/ft_ernie-pixel-clm_mnli_pair.sh
```


## XNLI
we fine-tuning pixelgpt on XNLI with two settings: `Translate-train-all` and `Cross-lingaul transfer`.
### Translate-train-all
#### ernie-clm-base
```
bash run/cross_lingual/xnli/train_all/ernie-clm-base/ft_ernie-clm-base_xnli.sh
```
#### ernie-pixel-only
```
bash run/cross_lingual/xnli/train_all/ernie-pixel-only/ft_ernie-pixel-only_xnli.sh
```
#### ernie-pixel-mono
- text-only
```
bash run/cross_lingual/xnli/train_all/ernie-pixel-mono/ft_ernie-pixel-mono_xnli_text.sh
```
- pixel-only
```
bash run/cross_lingual/xnli/train_all/ernie-pixel-mono/ft_ernie-pixel-mono_xnli_image.sh
```
- pair
```
bash run/cross_lingual/xnli/train_all/ernie-pixel-mono/ft_ernie-pixel-mono_xnli_pair.sh
```
#### ernie-pixel-clm
- text-only
```
bash run/cross_lingual/xnli/train_all/ernie-pixel-clm/ft_ernie-pixel-clm_xnli_text.sh
```
- pixel-only
```
bash run/cross_lingual/xnli/train_all/ernie-pixel-clm/ft_ernie-pixel-clm_xnli_image.sh
```
- pair
```
bash run/cross_lingual/xnli/train_all/ernie-pixel-clm/ft_ernie-pixel-clm_xnli_pair.sh
```

### Cross-lingaul Transfer
#### ernie-clm-base
```
bash run/cross_lingual/xnli/train_en/ernie-clm-base/ft_ernie-clm-base_xnli.sh
```
#### ernie-pixel-only
```
bash run/cross_lingual/xnli/train_en/ernie-pixel-only/ft_ernie-pixel-only_xnli.sh
```
#### ernie-pixel-mono
- text-only
```
bash run/cross_lingual/xnli/train_en/ernie-pixel-mono/ft_ernie-pixel-mono_xnli_text.sh
```
- pixel-only
```
bash run/cross_lingual/xnli/train_en/ernie-pixel-mono/ft_ernie-pixel-mono_xnli_image.sh
```
- pair
```
bash run/cross_lingual/xnli/train_en/ernie-pixel-mono/ft_ernie-pixel-mono_xnli_pair.sh
```
#### ernie-pixel-clm
- text-only
```
bash run/cross_lingual/xnli/train_en/ernie-pixel-clm/ft_ernie-pixel-clm_xnli_text.sh
```
- pixel-only
```
bash run/cross_lingual/xnli/train_en/ernie-pixel-clm/ft_ernie-pixel-clm_xnli_image.sh
```
- pair
```
bash run/cross_lingual/xnli/train_en/ernie-pixel-clm/ft_ernie-pixel-clm_xnli_pair.sh
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
