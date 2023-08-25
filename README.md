# Lil Bevo &mdash; UT Austin's submission to BabyLM Challenge

This repository contains code and instructions to build Lil Bevo &mdash; UT Austin's submission towards the [BabyLM Challenge](https://babylm.github.io).

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
python training_bevo.py --config_name microsoft/deberta-v3-small --tokenizer_name_or_path tokenizers/10m_maestro/ --train_file babylm_data/maestro/all-10M.txt --validation_file babylm_data/babylm_dev/dev.txt --per_device_train_batch_size 770 --per_device_eval_batch_size 128 --do_train --num_train_epochs 5 --do_eval --save_strategy epoch --optim adamw_torch_fused --warmup_ratio=0.0001 --weight_decay 0.1 --log_level error --learning_rate 5e-4 --evaluation_strategy steps --eval_steps 500 --output_dir deberta-small/redux/ --logging_steps 10 --save_total_limit 1 --overwrite_output_dir --torch_compile True --disable_tqdm False --max_seq_length 32 --report_to wandb
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

## Training Regime for Lil-Bevo

1. 5 epochs on MAESTRO dataset (85M non-language music tokens) combined with strict small dataset.
2. 50 epochs of pretraining with sequence length of 128 on strict-small dataset.
3. 2 epochs of targeted MLM.

## Training Regime for Lil-Bevo-X

1. 5 epochs on MAESTRO dataset (85M non-language music tokens) combined with strict small dataset.
2. 50 epochs of pretraining with sequence length of 128 on strict dataset.
3. 150 epochs of pretraining with sequence length of 512 on strict dataset.
4. 10 epochs of targeted MLM.

Please read [our paper]() to get more details on our training regime and reasoning behind these decisions.

## Results

*DynaBench*

| Model | Score |
| --- | --- | 
| Lil-Bevo | 0.70 |
| Lil-Bevo-X | 0.73 | 

*BLiMP*
| Model | Anaphor Agr. | Agr. Structure | Binding | Control/Raising | D-N Agr. | Ellipsis | Filler-Gap | Irregular Forms | Island Effects | NPI Licensing | Quantifiers | S-V Agr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lil-Bevo | 90.9 | 72.5 | 63.3 | 70.0 | 91.7 | 82.0 | 77.5 | 85.3 | 55.8 | 78.5 | 68.7 | 84.8 | 
| Lil-Bevo-X | 97.2 | 80.6 | 63.9 | 69.5 | 96.4 | 87.0 | 78.4 | 89.2 | 71.4 | 85.6 | 63.2 | 86.3 |


*BLiMP Supplement*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking |
| --- | --- | --- | --- | --- | --- |
| Lil-Bevo | 48.1 | 82.8 | 57.0 | 76.5 | 68.2 |
| Lil-Bevo-X | 45.2 | 75.0 | 63.6 | 81.4 | 78.2 |

*(Super)GLUE*
| Model | CoLA | SST-2 | MRPC (F1) | QQP (F1) | MNLI | MNLI-mm | QNLI | RTE | BoolQ | MultiRC | WSC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lil-Bevo | 73.7 | 88.4 | 82.2 | 85.5 | 75.4 | 76.3 | 81.6 | 46.5 | 65.4 | 66.0 | 61.5 |
| Lil-Bevo-X| 76.5 | 88.8 | 82.6 | 86.4 | 77.7 | 79.0 | 83.6 | 49.5 | 68.0 | 65.6 | 61.4 |

*MSGS*
| Model | CR (Control) | CR_LC | CR_RTP | LC (Control) | MV (Control) | MV_LC | MV_RTP | RP (Control) | SC (Control) | SC_LC | SC_RP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lil-Bevo | 91.9 | 66.6 | 67.4 | 100.0 | 99.8 | 75.7 | 78.0 | 93.8 | 91.5 | 65.7 | 64.2 |
| Lil-Bevo-X | 92.5 | 66.5 | 68.5 | 100.0 | 100.0 | 66.7 | 68.5 | 99.1 | 90.0 | 68.2 | 64.7 | 

*Age-of-acquisition Prediction*
(Mean absolute deviation in months across LOO cross-validation folds)
| Model | Overall (591 words) | Nouns (322) | Predicates (167) | Function words (102) |
| --- | --- | --- | --- | --- |
| Lil-Bevo | 2.06 | 2.0 | 1.84 | 2.65 |
| Lil-Bevo-X |  |  |  |  |
