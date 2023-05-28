'''
This script takes the saved pytorch checkpoints and makes a Huggingface based model for evaluation and submission.
'''

import torch
import transformers
import tokenizers
import ipdb
import sentencepiece
from model import BevoConfig, Bevo
from transformers import LlamaTokenizer, AutoModelForCausalLM

if __name__=="__main__":
    
    ########## TOKENIZER ############
    tokenizer = LlamaTokenizer(
        'tokenizers/babylm_10m_uni_16k.model',
        pad_token="<pad>",
        add_bos_token=False,
        add_eos_token=True,
    )
    
    ########## MODEL #############
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
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    device = 'cpu' # macbook mps is possible with a different environment setup I believe
    # system
    dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False # use PyTorch 2.0 to compile the model to be faster
    
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout)
    
    checkpoint = torch.load('out/ckpt.pt', map_location='cpu')
    
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
    
    ipdb.set_trace()
