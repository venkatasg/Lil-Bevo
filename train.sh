#!/bin/bash -x
#SBATCH -p gpu-a100
#SBATCH -J opt-125m
#SBATCH -o out/opt-125m.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user=venkatasg@utexas.edu
#SBATCH --mail-type=all

module reset

cd $WORK
source miniconda3/etc/profile.d/conda.sh
conda activate bevo
cd babylm/

# Script for training_bevo.py
torchrun --standalone --nproc_per_node=3 train.py --data babylm_data/babylm_10M/ --tokenizer_model_path tokenizers/babylm_10m_uni_16k.model --wandb_log --wandb_run_name 10Mtoks_100Mparams_16k_baseline_2048 --out_dir 10Mtoks_100Mparams_16k_baseline_2048 --seq_len 2048

python training_decoder.py --config_name facebook/opt-125m --tokenizer_name tokenizers/babylm_10m_uni_16k.model --train_file babylm_data/babylm_10M/train.txt --validation_file babylm_data/babylm_dev/dev.txt --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --num_train_epochs 10 --do_eval --logging_steps 0.02 --logging_first_step True --eval_steps 0.1 --max_eval_samples 5000 --save_steps 0.1 --evaluation_strategy steps --output_dir opt-125m-16k-10epochs --report_to wandb --run_name opt-125m-16k-10epochs --overwrite_output_dir 

python training_encoder.py --config_name microsoft/deberta-v3-base --tokenizer_name tokenizers/babylm_10m_uni_16k.model --train_file babylm_data/babylm_10M/train.txt --validation_file babylm_data/babylm_dev/dev.txt --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --num_train_epochs 20 --do_eval --logging_steps 0.02 --logging_first_step True --eval_steps 0.1 --max_eval_samples 5000 --save_steps 1 --evaluation_strategy steps --output_dir deberta-16k-10epochs --report_to wandb --run_name deberta-16k-20epochs --overwrite_output_dir 
