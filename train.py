"""
Training script copied from Karpathy's nanoGPT

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import time
import math
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import ipdb
from transformers import T5Tokenizer, set_seed, AutoModelForCausalLM, logging as logging_transformers
import logging
import random
from glob import glob
from datasets import Dataset, load_from_disk, logging as logging_datasets
from random import shuffle
import sys

from model import BevoConfig, BevoForCausalLM

logging_datasets.disable_progress_bar()
logging_transformers.set_verbosity_error()
logger = logging.getLogger("pytorch")
logger.propagate = False
logger.setLevel(logging.ERROR)

def arg_parse():
    # Define argument parser instead of the weird configurator that Karpathy setup
    parser = argparse.ArgumentParser(
    prog='Training script for lil Bevo',
    description='Trains a GPT style decoder model on training data',
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True, 
        help="point to babyLM 10M or 100M data directory."
    )
    parser.add_argument(
        '--tokenizer_model_path',
        type=str,
        required=True, 
        help="point to tokenizer .model file to use"
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='out/',
        help="where to store model outputs"
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=128,
        help="Maximum sequence length (in tokens) for input."
    )
    parser.add_argument(
        '--num_hidden_layers',
        type=int,
        default=12,
        help="Number of transformer blocks."
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help="Embedding size."
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=2048,
        help="Positional encoding block size."
    )
    parser.add_argument(
        '--num_attention_heads',
        type=int,
        default=12,
        help="Number of attention_heads."
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help="Dropout in MLP layers."
    )
    parser.add_argument(
        '--attention_dropout',
        type=float,
        default=0,
        help="Dropout in attention layers."
    )
    parser.add_argument(
        '--bias',
        type=bool,
        default=True,
        help="Bias in hidden layers."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=12,
        help="Batch size."
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='How often to log metrics'
    )
    parser.add_argument(
        '--eval_interval',
        type=int,
        default=100,
        help="how often to run eval"
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help="Pass if you just want to evaluate the model"
    )
    parser.add_argument(
        '--always_save_checkpoint',
        action='store_true',
        help="Always save checkpoint regardless of whether validation loss goes down"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda', 
        help="cpu, cuda, mps"
    )
    parser.add_argument(
        '--init_from',
        type=str,
        default='scratch', 
        help="scratch or resume"
    )
    parser.add_argument(
        '--wandb_log',
        action='store_true',
        help="Log to wandb"
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='lil-bevo', 
        help="Don't change this"
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        help="You must set a run name if wandb_log is passed"
    )
    args = parser.parse_args()
    
    return args

if __name__=="__main__":
    args = arg_parse()

    # -----------------------------------------------------------------------------
    # Hyperparameters for training
    gradient_accumulation_steps = 6 * 6 # used to simulate larger batch sizes
    
    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    max_iters = 100000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    device = args.device
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True if args.device!='mps' else False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        assert gradient_accumulation_steps % torch.cuda.device_count() == 0
        gradient_accumulation_steps //= torch.cuda.device_count()
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * args.batch_size * args.block_size
    # print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(78727 + seed_offset)
    torch.cuda.manual_seed_all(78727 + seed_offset)
    random.seed(78727 + seed_offset)
    np.random.seed(78727 + seed_offset)
    set_seed(78727)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    #-------------------------------------------------------------------------------
    # Data loading, tokenization, etc
    
    def split_sequence_gen(seq_len):
        def split_sequence(batch):
            concatenated_examples = {k: sum(batch[k], []) for k in batch.keys()}
            total_length = len(concatenated_examples[list(batch.keys())[0]])
            # Find what length to add padding
            remainder = total_length % seq_len
            if remainder != 0:
                to_add = seq_len - remainder
            elif remainder == 0:
                to_add = 0
            to_add_input_id = [tokenizer.pad_token_id] * to_add
            to_add_atten_mask = [0] * to_add
            
            # split at 128 and pad the rest
            pad_dict = dict(input_ids=to_add_input_id, attention_mask=to_add_atten_mask)
            for key in concatenated_examples.keys():
                t = concatenated_examples[key]
                t1 = [item for sublist in [t, pad_dict[key]] for item in sublist]
                assert not len(t1) % seq_len
                concatenated_examples[key] = t1
            total_length_use = len(concatenated_examples[list(batch.keys())[0]])
            result = {
                k: [t[i: i + seq_len] for i in range(0, total_length_use, seq_len)]
                for k, t in concatenated_examples.items()
            }
            
            # Label is -100 if attention mask is 0, otherwise same as input ids
            result["labels"] = result["input_ids"].copy()
            result["labels"] = [
                [tokenizer.eos_token_id if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in
                [zip(masks, labels) for masks, labels in zip(result["attention_mask"], result["labels"])]
            ]
            
            # Some checks
            assert all([len(x) == seq_len for x in result["input_ids"]])
            assert all([len(x) == seq_len for x in result["attention_mask"]])
            assert all([len(x) == seq_len for x in result["labels"]])
            return result
        return split_sequence
    
    def load_data(tokenizer, data_path):
        '''
        This loads the data, tokenizes it and makes a pyTorch dataloader of it
        
        RETURNS
        pyTorch Dataloader
        '''
        
        # Load the text, and split into array of strings
        files = glob(data_path + '*')
        
        # Load the data and make into a datasets object
        all_lines = []
        for file in files:
            # exclude directories
            if ('.train' in file) or ('.dev' in file) or ('.test' in file):
                with open(file, 'r', encoding="utf-8") as f:
                    all_lines.extend(f.readlines())
                    
        all_data = [{'text': x.strip()} if x.strip() else {'text': ""} for x in all_lines]
        
        # Convert all_data to huggingface datasets object
        full_dataset = Dataset.from_list(all_data)
        full_dataset = full_dataset.map(
            lambda x: tokenizer(
                x['text']
            ),
            remove_columns=["text"],
            batched=True
        )
        
        # Splits the data into 128 token length sequences with padding
        full_dataset = full_dataset.map(
            split_sequence_gen(args.seq_len), 
            batched=True
        )
        
        full_dataset.set_format("pt", columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
        
        dataloader = DataLoader(full_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=True
        )
        return dataloader
        
    train_path = args.data_dir
    val_path = '/'.join(args.data_dir.split('/')[:-2]) + '/babylm_dev/'
    
    # This needs to point to a directory I think
    tokenizer = T5Tokenizer(
        args.tokenizer_model_path
    )
        
    train_dataloader = load_data(tokenizer, train_path)
    if master_process:
        print("Loaded training data")
    
    val_dataloader = load_data(tokenizer, val_path)
    if master_process:
        print("Loaded validation data")
    
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
        
    #-------------------------------------------------------------------------------
    # model init
    
    model_args = dict(num_hidden_layers=args.num_hidden_layers, num_attention_heads=args.num_attention_heads, hidden_size=args.hidden_size, block_size=args.block_size, bias=args.bias, vocab_size=None, dropout=args.dropout, attention_dropout=args.attention_dropout) # start with model_args from command line
    if args.init_from == 'scratch':
        # init a new model from scratch
        if master_process:
            print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        model_args['vocab_size'] =  tokenizer.vocab_size
        config = BevoConfig(**model_args)
        model = BevoForCausalLM(config)
    elif args.init_from == 'resume':
        if master_process:
            print(f"Resuming training from {args.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['num_hidden_layers', 'num_attention_heads', 'hidden_size', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        config = BevoConfig(**model_args)
        model = BevoForCausalLM(config).from_pretrained(args.out_dir)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        
    # Register the model
    BevoConfig.register_for_auto_class()
    BevoForCausalLM.register_for_auto_class('AutoModelForCausalLM')
    
    # crop down the model block size if desired, using model surgery
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
        model_args['block_size'] = args.block_size # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if args.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory
    
    # compile the model
    if compile:
        if master_process:
            print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    
    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            eval_iters = 100 if split=='train' else len(val_dataloader.dataset)//args.batch_size
            dataloader = train_dataloader if split=='train' else val_dataloader
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                with ctx:
                    batch = next(iter(dataloader))
                    outputs = model(batch['input_ids'].to(device), batch['labels'].to(device))
                losses[k] = outputs['loss'].item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    
    # logging
    if args.wandb_log and master_process:
        import wandb
        if args.wandb_run_name is None:
            sys.exit("Pass a meaningful wandb_run_name argument!")
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name, 
            config=config
        )
    
    # report number of parameters
    if master_process:
        print("number of parameters: %.2fM" % (model.get_num_params()/1e6,))
    
    max_iters = len(train_dataloader.dataset)//args.batch_size
    lr_decay_iters = max_iters
    if master_process:
        print("Setting max iters to total number of batches: ", max_iters)
        
    # Training loop
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if args.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            if losses['val'] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {args.out_dir}")
                    raw_model.save_pretrained(args.out_dir)
                    tokenizer.save_pretrained(args.out_dir)
                    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
        if iter_num == 0 and args.eval_only:
            break
            
        # Get first batch of training data
        batch = next(iter(train_dataloader))
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                outputs = model(batch['input_ids'].to(device), batch['labels'].to(device))
                logits, loss = outputs['logits'], outputs['loss']
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            batch = next(iter(train_dataloader))
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(args.batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1
        
        # termination conditions
        if iter_num >= max_iters:
            break
    
    if ddp:
        destroy_process_group()
