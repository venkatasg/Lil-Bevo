"""
Training script copied from Karpathy's nanoGPT

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
import argparse
import ipdb
from transformers import LlamaTokenizer, set_seed
import logging
import random
from glob import glob
from datasets import Dataset, load_from_disk
from random import shuffle
import sys

from model import BevoConfig, Bevo

# logging.set_verbosity_error()
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
        help="where to model"
    )
    parser.add_argument(
        '--long_sequence_strat',
        type=str,
        default='split',
        help="How to handle long sequences: split or sample and concat"
    )
    parser.add_argument(
        '--seq_length',
        type=int,
        default=128,
        help="Maximum sequence length (in tokens) for input."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=12,
        help="Batch size."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help="Number of dataloader workers."
    )
    parser.add_argument(
        '--eval_interval',
        type=int,
        default=2000,
        help="How often to run eval while training"
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='How often to log metrics'
    )
    parser.add_argument(
        '--eval_iters',
        type=int,
        default=200,
        help=""
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
    block_size = 1024
    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
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
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    device = args.device
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
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
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * args.batch_size * block_size
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
    
    def sample_sequence_gen(seq_length, eos_token_id):
        def sample_sequence(line):
            doc_length = line["input_ids"].shape[0]
            if doc_length <= seq_length:
                start = 0
            else:
                if random.random() < 1 / 4:
                    start = 0
                else:
                    start = random.randint(0, doc_length - seq_length)
            input_ids = line["input_ids"][start : start + seq_length]
            # if input_ids[-1] != eos_token_id:
            #     input_ids[-1] = eos_token_id
            return {"input_ids": input_ids}
        return sample_sequence
    
    def split_sequence_gen(seq_length):
        def split_sequence(batch):
            input_ids = batch["input_ids"][0]
            out = []
            while len(input_ids) >= (1 + len(out)) * seq_length:
                out.append(input_ids[len(out) * seq_length : (1 + len(out)) * seq_length])
            return {"input_ids": out}
        return split_sequence
    
    def concat_multiple_sequence_gen(seq_length, pad_token_id):
        def concat_multiple_sequence(batch):
            concat_input_ids = torch.cat([batch['input_ids'][i] for i in range(len(batch['input_ids']))], dim=0)
            # remove all pad tokens
            concat_input_ids = concat_input_ids[concat_input_ids!=pad_token_id]
            length = concat_input_ids.shape[0]
            chunks = math.ceil(length / seq_length)
            pad_length = chunks * seq_length - length
            pad = torch.ones(pad_length, dtype=concat_input_ids.dtype) * pad_token_id
            # add pad tokens back
            concat_input_ids = torch.cat([concat_input_ids, pad], dim=0)
            # then break up into chunks. But how to handle sequence breaks at max length?
            input_ids = torch.chunk(concat_input_ids, chunks)
            input_ids = torch.stack(input_ids, dim=0)
            # if chunks > 1:
            #     ipdb.set_trace()
            return {"input_ids": input_ids}
            
        return concat_multiple_sequence
    
    def get_labels_gen(pad_token_id):
        def get_labels(line):
            input_ids = line["input_ids"]
            labels = input_ids.clone()
            # Shift one token to right to get labels and pad to 128
            labels = torch.cat([labels[1:], torch.ones(1, dtype=labels.dtype) * pad_token_id])
            return {"labels": labels}  
        return get_labels
    
    def load_data(tokenizer, data_path):
        '''
        This loads the data, tokenizes it and makes a pyTorch dataloader of it
        
        RETURNS
        pyTorch Dataloader
        '''
        
        # Load the text, and split into array of strings
        files = glob(data_path + '*')
        if (data_path + 'dataset-' + args.long_sequence_strat + '-' + args.tokenizer_model_path.split('/')[-1]) not in files:
            # Load the data and make into a datasets object
            all_data = ''
            for file in files:
                # exclude directories
                if ('.train' in file) or ('.dev' in file) or ('.test' in file):
                    with open(file, 'r') as f:
                        all_data += f.read()
                    
            all_data = [{'text': x} for x in all_data.split('\n')]
            shuffle(all_data)
            
            # Convert all_data to huggingface datasets object
            full_dataset = Dataset.from_list(all_data)
            full_dataset = full_dataset.map(
                lambda x: tokenizer(
                    x['text'], 
                    padding='max_length', 
                    max_length=args.seq_length, 
                    return_tensors='pt',
                    truncation='do_not_truncate',
                    return_length=False,
                    return_attention_mask=False
                )
            )
            
            full_dataset = full_dataset.map(lambda x: {"input_ids": x["input_ids"][0]})
            full_dataset = full_dataset.select_columns("input_ids")
            full_dataset.set_format("pt", columns=["input_ids"], output_all_columns=True)
            if args.long_sequence_strat=='split':
                # This just splits the document into 128 token length sequences
                full_dataset = full_dataset.map(
                    split_sequence_gen(args.seq_length), batched=True, batch_size=1
                )
            elif args.long_sequence_strat=='sample':
                # This does some fancy sampling from gopher paper and reduces padding
                full_dataset = full_dataset.map(
                    sample_sequence_gen(args.seq_length, tokenizer.eos_token_id)
                )
                full_dataset = full_dataset.map(
                    concat_multiple_sequence_gen(args.seq_length, tokenizer.pad_token_id),
                    batched=True,
                    batch_size=10,
                    drop_last_batch=True,
                )
            full_dataset = full_dataset.select_columns("input_ids")
            # Get labels: which is just the next token to predict
            full_dataset = full_dataset.map(get_labels_gen(tokenizer.pad_token_id))
            full_dataset.save_to_disk(data_path + 'dataset-' + args.long_sequence_strat + '-' + args.tokenizer_model_path.split('/')[-1])
        else:
            full_dataset = load_from_disk(data_path + 'dataset-' + args.long_sequence_strat + '-' + args.tokenizer_model_path.split('/')[-1])
            
        dataloader = DataLoader(full_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True
            )
        
        return dataloader
        
    train_path = args.data_dir
    val_path = '/'.join(args.data_dir.split('/')[:-2]) + '/babylm_dev/'
    
    # This needs to point to a directory I think
    tokenizer = LlamaTokenizer(
        args.tokenizer_model_path,
        pad_token="<pad>",
        add_bos_token=False,
        add_eos_token=True,
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
    
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    if args.init_from == 'scratch':
        # init a new model from scratch
        if master_process:
            print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        model_args['vocab_size'] =  tokenizer.vocab_size
        model_config = BevoConfig(**model_args)
        model = Bevo(model_config)
    elif args.init_from == 'resume':
        if master_process:
            print(f"Resuming training from {args.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        model_config = BevoConfig(**model_args)
        model = Bevo(model_config)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
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
            dataloader = train_dataloader if split=='train' else val_dataloader
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                with ctx:
                    batch = next(iter(dataloader))
                    logits, loss = model(batch['input_ids'].to(device), batch['labels'].to(device))
                losses[k] = loss.item()
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
            config=model_config
        )
    if master_process:
        print("Total number of batches: ", len(train_dataloader.dataset)//args.batch_size)
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
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': model_config,
                    }
                    print(f"saving checkpoint to {args.out_dir}")
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
                logits, loss = model(batch['input_ids'].to(device), batch['labels'].to(device))
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
        if iter_num > max_iters:
            break
    
    if ddp:
        destroy_process_group()
