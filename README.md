# Lil Bevo &mdash; UT Austin's submission to BabyLM Challenge

This repository contains Lil Bevo &mdash; UT Austin's submission towards the [BabyLM Challenge](https://babylm.github.io).

## Python Environment

Install latest version of `miniconda` from [here](https://docs.conda.io/en/latest/miniconda.html).

The exact python environment configuration is stored in `environment.yml`. To recreate it in `conda`:

```
conda env create -f environment.yml
``` 

This will create an environment called `bevo` with all required packages. `CUDA` support might need to be configured separately depending on the machine you're running it on.

Alternatively, run the following commands in order:

```
conda create -n bevo pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia jupyter pandas numpy matplotlib scikit-learn tqdm
pip install git+https://github.com/huggingface/transformers wandb ipdb datasets sentencepiece evaluate pytest accelerate mido
```

## Scripts

`training_bevo.py` and `modeling_bevo.py` are based on `train.py` and `model.py` from Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT/), with mostly the same hyperparameters (except block_size = seq_len is 128 instead of 1024, and bias set to True).
~~`training_bevo.py` and `modeling_bevo.py` still have bugs in them.~~ (I think the bugs are gone but proceed with caution). 

Example run:

```
python training_bevo.py --data babylm_data/babylm_10M/ --tokenizer_model_path tokenizers/babylm_100m_uni_16k.model --dropout 0.0 --wandb_run_name some_name_for_run --out_dir name_of_output_dir --wandb_log
```

`training_decoder.py` takes as argument any decoder style LM on the Huggingface Hub, and trains the model on babyLM data. First, concatenate all the train and dev files into one text file to pass as input to this script (`cat babylm_data/babylm_10M/*.train > train.txt`). Set the `WANDB_PROJECT` environment variable to **lil-bevo** and run.

```
export WANDB_PROJECT="lil-bevo"
python training_decoder.py --config_name facebook/opt-125m --tokenizer_name tokenizers/babylm_10m_uni_16k.model --train_file babylm_data/babylm_10M/train.txt --validation_file babylm_data/babylm_dev/dev.txt --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --num_train_epochs 10 --do_eval --logging_steps 0.02 --logging_first_step True --eval_steps 0.1 --max_eval_samples 5000 --save_steps 1 --evaluation_strategy steps --output_dir opt-125m-16k-10epochs --report_to wandb --run_name opt-125m-16k-10epochs --overwrite_output_dir 
```

`training_encoder.py` is the same --- it takes as argument any encoder style LM from the Huggingface hub.

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

Before running the benchmark, comment out line 163 in file 'lm_eval/models/huggingface.py'. I believe this is a bug in their code.

To run the BLiMP benchmark:

```
python babylm_eval.py PATH_TO_SAVED_MODEL 'decoder' --trust_remote_code
```

## Baseline Models

Models in bold are ones that we trained and got results.

**Strict-small Track: 10M tokens**

*BLiMP*
| Model | Anaphor Agr. | Agr. Structure | Binding | Control/Raising | D-N Agr. | Ellipsis | Filler-Gap | Irregular Forms | Island Effects | NPI Licensing | Quantifiers | S-V Agr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DeBERTa-best** | 88.0 | 69.0 | 67.9 | 66.6 | 88.7 | 82.5 | 67.1 | 85.0 | 42.5 | 58.2 | 64.8 | 73.1 |
| **OPT-125m-best** | 87.3 | 67.2 | 67.7 | 63.9 | 84.4 | 56.1 | 64.5 | 88.2 | 45.0 | 51.2 | 74.3 | 64.4 |
| OPT-125m | 63.8 | 70.6 | 67.1 | 66.5 | 78.5 | 62 | 63.8 | 67.5 | 48.6 | 46.7 | 59.6 | 56.9 |
| RoBERTa-base | 81.5 | 67.1 | 67.3 | 67.9 | 90.8 | 76.4 | 63.5 | 87.4 | 39.9 | 55.9 | 70.5 | 65.4 |
| T5-base | 68.9 | 63.8 | 60.4 | 60.9 | 72.2 | 34.4 | 48.2 | 77.6 | 45.6 | 47.8 | 61.2 | 65.0 |


## Music

### MAESTRO

We attempted pretraining with the [MAESTRO]() dataset, in addition to the 10M tokens in the training data, to see if hierarchical information in music might help. Convert all the midi files into one big text file (split every 1000 spaces) using `midi_to_text.py`. Then train a tokenizer on the combined file:

```
python midi/midi_to_text.py babylm_data/maestro/*/*.midi > sed 's/ /\n/1000; P; D' > babylm_data/maestro/midi.txt
 cat babylm_data/babylm_10M/all.txt babylm_data/maestro/midi.txt
 > babylm_data/maestro/all.txt
```

The following results are for a tokenizer trained on the 10M tokens + MAESTRO, with similar pretraining paradigm in all 3 cases.

**Strict-small Track: 10M tokens**

*BLiMP*
| Model | Anaphor Agr. | Agr. Structure | Binding | Control/Raising | D-N Agr. | Ellipsis | Filler-Gap | Irregular Forms | Island Effects | NPI Licensing | Quantifiers | S-V Agr. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **OPT** | 83.13 | 68.5 | 67.0 | 64.3 | 82.9 | 62.4 | 64.6 | 88.3 | 49.8 | 51.2 | 72.2 | 64.4 |
| **OPT+maestro** | 83.3 | 68.0 | 66.8 | 64.0 | 83.0 | 60.4 | 65.1 | 87.6 | 50.0 | 53.0 | 69.3 | 63.8 |
| OPT-125m-babyLM-baseline | 63.8 | 70.6 | 67.1 | 66.5 | 78.5 | 62 | 63.8 | 67.5 | 48.6 | 46.7 | 59.6 | 56.9 |
