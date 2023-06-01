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
# torchrun --standalone --nproc_per_node=3 train.py --data babylm_data/babylm_10M/ --tokenizer_model_path tokenizers/babylm_10m_uni_16k.model --wandb_log --wandb_run_name 10Mtoks_100Mparams_16k_baseline_2048 --out_dir 10Mtoks_100Mparams_16k_baseline_2048 --seq_len 2048

# Script for training_opt.py
python training_decoder.py --config_name facebook/opt-125m --tokenizer_name tokenizers/babylm_10m_uni_16k.model --train_file babylm_data/babylm_10M/train.txt --validation_file babylm_data/babylm_dev/dev.txt --per_device_train_batch_size 12 --per_device_eval_batch_size 32 --do_train --num_train_epochs 3 --do_eval --logging_steps 10 --eval_steps 0.1 --max_eval_samples 2000 --evaluation_strategy steps --gradient_accumulation_steps 12 --output_dir opt-125m-16k-3epoch --report_to wandb --run_name opt-125m-16k-3epoch --overwrite_output_dir 
