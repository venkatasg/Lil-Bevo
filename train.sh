#!/bin/bash -x
#SBATCH -J babylm-train
#SBATCH -o babylm-train.txt
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-user=venkatasg@utexas.edu
#SBATCH --mail-type=all

module reset

cd $WORK
source miniconda3/etc/profile.d/conda.sh
conda activate bevo
cd babylm/

torchrun --standalone --nproc_per_node=3 train.py --data babylm_data/babylm_10M/ --tokenizer_model_path tokenizers/babylm_10m_uni_16k.model --wandb_log --wandb_run_name test_run_100M_16k_no_stop_token_hf --eval_interval 200 --out_dir 100Mbaseline_16k

torchrun --standalone --nproc_per_node=3 train.py --data babylm_data/babylm_10M/ --tokenizer_model_path tokenizers/babylm_10m_uni_32k.model --wandb_log --wandb_run_name test_run_100M_32k_no_stop_token_hf --eval_interval 200 --out_dir 100Mbaseline_32k
