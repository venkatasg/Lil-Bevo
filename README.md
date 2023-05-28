# Lil Bevo &mdash; UT Austin's submission to BabyLM Challenge

This repository contains Lil Bevo &mdash; UT Austin's submission towards the [BabyLM Challenge](https://babylm.github.io).

## Python Environment

Install latest version of `miniconda` from [here](https://docs.conda.io/en/latest/miniconda.html).

The exact python environment configuration is stored in `environment.yml`. To recreate it in `conda`:

```
conda env create -f environment.yml
``` 

This will create an environment called `bevo` with all required packages. `CUDA` support might need to be configured separately depending on the machine you're running it on.

Alternatively, run the following install commands in order:

```
conda create -n bevo pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia jupyter pandas numpy matplotlib scikit-learn tqdm
pip install git+https://github.com/huggingface/transformers
pip install wandb ipdb datasets sentencepience
```

## Evaluation

Setup evaluation pipeline as [the BabyLM repo instructs](https://github.com/babylm/evaluation-pipeline)

### Baseline model

Our baseline model is based on [nanoGPT](https://github.com/karpathy/nanoGPT/) by Andrej Karpathy. The model has **97.6M** parameters, so should be comparable to OPT-125M.

**Strict-small Track**

*BLiMP*
| Model | Anaphor Agr. | Agr. Structure | Binding | Control/Raising | D-N Agr. | Ellipsis | Filler-Gap | Irregular Forms | Island Effects | NPI Licensing | Quantifiers | S-V Agr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline** |  |  |  |  |  |  |  |  |  |  |  |  |
| OPT-125m | 63.8 | 70.6 | 67.1 | 66.5 | 78.5 | 62 | 63.8 | 67.5 | 48.6 | 46.7 | 59.6 | 56.9 |
| RoBERTa-base | 81.5 | 67.1 | 67.3 | 67.9 | 90.8 | 76.4 | 63.5 | 87.4 | 39.9 | 55.9 | 70.5 | 65.4 |
| T5-base | 68.9 | 63.8 | 60.4 | 60.9 | 72.2 | 34.4 | 48.2 | 77.6 | 45.6 | 47.8 | 61.2 | 65.0 |

*(Super)GLUE*
| Model | CoLA | SST-2 | MRPC (F1) | QQP (F1) | MNLI | MNLI-mm | QNLI | RTE | BoolQ | MultiRC | WSC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline** |  |  |  |  |  |  |  |  |  |  |  |  |
| *Majority label* | *69.5* | *50.2* | *82* | *53.1* | *35.7* | *35.7* | *35.4* | *53.1* | *50.5* | *59.9* | *53.2* | *61.4* |
| OPT-125m | 64.6 | 81.9 | 72.5 | 60.4 | 57.6 | 60.0 | 61.5 | 60.0 | 63.3 | 55.2 | 60.2 |
| RoBERTa-base | 70.8 | 87.0 | 79.2 | 73.7 | 73.2 | 74.0 | 77.0 | 61.6 | 66.3 | 61.4 | 61.4 |
| T5-base | 61.2 | 78.1 | 80.5 | 66.2 | 48.0 | 50.3 | 62.0 | 49.4 | 66.0 | 47.1 | 61.4 |
