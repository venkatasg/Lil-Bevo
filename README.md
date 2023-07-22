# Lil Bevo &mdash; UT Austin's submission to BabyLM Challenge

This repository contains Lil Bevo &mdash; UT Austin's submission towards the [BabyLM Challenge](https://babylm.github.io).

## Python Environment

Install latest version of `miniconda` from [here](https://docs.conda.io/en/latest/miniconda.html).

To recreate the exact python environment configuration in `conda`, run the following commands in order:

```
conda create -n bevo pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia jupyter pandas numpy matplotlib scikit-learn tqdm
pip install git+https://github.com/huggingface/transformers wandb ipdb datasets sentencepiece evaluate pytest accelerate mido
```

## Scripts

`training_bevo.py` takes as argument any decoder style LM on the Huggingface Hub, and trains the model on babyLM data. First, concatenate all the train and dev files into one text file to pass as input to this script (`cat babylm_data/babylm_10M/*.train > train.txt`). Set the `WANDB_PROJECT` environment variable to **lil-bevo** and run.

```
export WANDB_PROJECT="lil-bevo"
python training_decoder.py --config_name facebook/opt-125m --tokenizer_name tokenizers/babylm_10m_uni_16k.model --train_file babylm_data/babylm_10M/train.txt --validation_file babylm_data/babylm_dev/dev.txt --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --num_train_epochs 10 --do_eval --logging_steps 0.02 --logging_first_step True --eval_steps 0.1 --max_eval_samples 5000 --save_steps 1 --evaluation_strategy steps --output_dir opt-125m-16k-10epochs --report_to wandb --run_name opt-125m-16k-10epochs --overwrite_output_dir 
```

## Evaluation

To setup evaluation pipeline as [the BabyLM repo instructs](https://github.com/babylm/evaluation-pipeline), but in a separate conda environment:

```
git clone https://github.com/babylm/evaluation-pipeline
cd evaluation-pipeline
conda create -n babyeval python==3.9 pip git-lfs
conda activate babyeval
pip install --no-build-isolation -e ".[dev]"
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 sentencepiece

```

## Models

We trained two models, one for the strict-small track and another for the strict track:

- [Lil-Bevo](https://huggingface.co/venkatasg/lil-bevo) is based on a `deberta-small-v3` model, and has 55M parameters with a vocab size of 16640.
- [Lil-Bevo-X](https://huggingface.co/venkatasg/lil-bevo-x) is based on a `deberta-base-v3` model and has 112M parameters with a vocab size of 33280.

## Training Regime

## Results

Results for our models are presented below, with baseline results. Lil-Bevo results are for the strict-small track(10M tokens), while Lil-Bevo-X are for the strict track(100M tokens).

*BLiMP*
| Model | Anaphor Agr. | Agr. Structure | Binding | Control/Raising | D-N Agr. | Ellipsis | Filler-Gap | Irregular Forms | Island Effects | NPI Licensing | Quantifiers | S-V Agr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lil-Bevo | 87.8 | 74.5 | 63.0 | 67.5 | 88.8 | 79.6 | 76.0 | 81.2 | 60.4 | 77.4 | 72.2 | 80.4 |
| Lil-Bevo-X | 96.7 | 81.3 | 65.7 | 72.5 | 95.2 | 82.5 | 82.2 | 88.9 | 70.6 | 83.4 | 59.9 | 88.9 |

*BLiMP Supplement*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking |
| --- | --- | --- | --- | --- | --- |
| Lil-Bevo | 48.6 | 68.8 | 57.0 | 81.2 | 62.9 |
| Lil-Bevo-X | 45.7 | 76.6 | 57.6 | 82.4 | 79.3 |

*(Super)GLUE*
| Model | CoLA | SST-2 | MRPC (F1) | QQP (F1) | MNLI | MNLI-mm | QNLI | RTE | BoolQ | MultiRC | WSC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lil-Bevo | 71.2 | 89.6 | 80.7 | 84.8 | 76.0 | 76.4 | 82.0 | 54.5 | 67.6 | 62.8 | 57.8 |
| Lil-Bevo-X| 76.5 | 90.8 | 82.4 | 87.6 | 78.1 | 80.7 | 85.7 | 53.5 | 67.2 | 64.8 | 61.4 |

*MSGS*
| Model | CR (Control) | LC (Control) | MV (Control) | RP (Control) | SC (Control) | CR_LC | CR_RTP | MV_LC | MV_RTP | SC_LC | SC_RP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lil-Bevo | 892.5 | 66.7 | 66.9 | 87.8 | 100.0 | 69.3 | 69.2 | 97.5 | 91.5 | 69.0 | 66.7 |
| Lil-Bevo-X | 90.0 | 68.1 | 66.9 | 100.0 | 100.0 | 66.8 | 84.5 | 99.3 | 95.4 | 68.4 | 64.9 | 

*Age-of-acquisition Prediction*
(Mean absolute deviation in months across LOO cross-validation folds)
| Model | Overall (591 words) | Nouns (322) | Predicates (167) | Function words (102) |
| --- | --- | --- | --- | --- |
| Lil-Bevo |  |  |  |  |
| Lil-Bevo-X |  |  |  |  |
