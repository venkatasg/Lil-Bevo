#!/bin/bash -x
#SBATCH -J babylm-eval
#SBATCH -o babylm-eval.txt
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user=venkatasg@utexas.edu
#SBATCH --mail-type=all

module reset

cd $WORK
source miniconda3/etc/profile.d/conda.sh
conda activate babyeval
cd babylm/evaluation-pipeline/

python babylm_eval.py '10Mtoks_100Mparams_16k_baseline' 'decoder'

# ./finetune_all_tasks.sh 'path/to/model_and_tokenizer'
# 
# ./collect_results.sh 'path/to/model_and_tokenizer'
