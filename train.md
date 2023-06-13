Script for training_bevo.py

```
torchrun --standalone --nproc_per_node=3 train.py --data babylm_data/babylm_10M/ --tokenizer_model_path tokenizers/babylm_10m_uni_16k.model --wandb_log --wandb_run_name 10Mtoks_100Mparams_16k_baseline_2048 --out_dir 10Mtoks_100Mparams_16k_baseline_2048 --seq_len 2048
```

Python script for training encoder/decoder style models from Huggingface hub:

```
python training_[encoder/decoder].py
    --model_name_or_path MODEL_NAME_OR_PATH 
    --tokenizer_name TOKENIZER_PATH
    --overwrite_cache
    --train_file TRAIN_TEXT_FILE 
    --validation_file VALIDATION_TEXT_FILE 
    --per_device_eval_batch_size 8 
    --do_train 
    --per_device_train_batch_size 8 
    --num_train_epochs 5 
    --warmup_ratio=0 
    --weight_decay 0.1 
    --learning_rate 5e-5
    --do_eval
    --evaluation_strategy steps 
    --eval_steps 0.05
    --max_eval_samples 5000
    --log_level error 
    --save_strategy epoch
    --save_total_limit 10 
    --torch_compile
    --ddp_backend nccl
    --output_dir OUTPUT_DIR
    --overwrite_output_dir  
    --report_to wandb 
    --run_name RUN_NAME 
    --disable_tqdm True
```
