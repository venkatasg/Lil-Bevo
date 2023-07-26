# https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt
"""
See modification in whole_word_masking_data_collator which replaces
instead of masking, using another generator model.

i.e., "simplified CLM" from "METRO: Efficient Denoising Pretraining of Large Scale Autoencoding
Language Models with Model Generated Signals"

TODO:
- multitask learning with added RTD loss.

"""
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import default_data_collator
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import logging as logging_datasets

import math
import torch
from datasets import Dataset, DatasetDict
import collections
import os
import numpy as np
from functools import partial
import wandb
from tqdm import tqdm
import torch.nn as nn

os.environ['WANDB_PROJECT']='lil-bevo'

logging_datasets.disable_progress_bar()
#NOTE: I may not need this after all, just change the data_collator
#NOTE: Currently unused, but will need this if doing multi-task.
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.loss_fn = nn.MSELoss()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer.
        By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.get("labels")  # or .pop?
        #input_ids = torch.tensor(inputs['input_ids'])
        outputs = model(**inputs)
        #outputs = model(input_ids = input_ids.unsqueeze(1))

        logits = outputs.get("logits")
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

        #return self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

def main():

    parser = argparse.ArgumentParser(description="Experiment with Corrective Language Modeling")

    parser.add_argument('-o', '--outputdirectory', type=str, help='Where to save trained model checkpoint.')
    parser.add_argument('-t', '--train_dir', type=str, help='train directory')
    parser.add_argument('-c', '--load_model', type=str, help='checkpoint to load')
    parser.add_argument('-s', '--from_scratch', action='store_true', help='Train deberta from scratch?')
    parser.add_argument('-e', '--epochs', type=int, help='Number of training epochs')
    parser.add_argument('-tb', '--per_device_train_batch_size', type=int, help='per device train Batch size')
    parser.add_argument('-lm', '--mlm_or_clm', type=str, help='Masked lm or corrective lm')
    args = parser.parse_args()

    outputdirectory = args.outputdirectory
    load_checkpoint_dir = args.load_model
    train_dir = args.train_dir
    from_scratch = args.from_scratch
    num_epochs = args.epochs
    data_collator_type = args.mlm_or_clm

    if data_collator_type not in ['mlm','clm','mlmfill']:
        raise ValueError("Must be mlm or clm or mlmfill")

    #NOTE: this is the 512- token version. For the 128-token version, see the other .txt files
    def load_original_and_replaced(train_dir, num_samples=None):
        print("Loading data...")

        with open(os.path.join(train_dir, "train-original.txt"),'r') as fd:
            traintext_orig = fd.readlines()
        with open(os.path.join(train_dir, "train-replaced.txt"),'r') as fd:
            traintext_repl = fd.readlines()

        if num_samples is not None:
            traintext_orig = traintext_orig[:num_samples]
            traintext_repl = traintext_repl[:num_samples]

        traindata = Dataset.from_dict({"origtext":traintext_orig, "repltext":traintext_repl})

        dataset = DatasetDict({"train": traindata} )
        print("Done loading data...")
        return dataset


    def tokenize_function2(tokenizer, examples):
        """ Load original and replaced versions of data first,
        to avoid having the data collator having to do the work!
        """
        max_seq_length = 512
        original = tokenizer(examples["origtext"],
                           truncation=True,
                           padding='max_length',
                           max_length = max_seq_length,
                        )
        replaced = tokenizer(examples["repltext"],
                           truncation=True,
                           padding='max_length',
                           max_length = max_seq_length,
                        )

        replaced["labels"] = original["input_ids"].copy()

        #TODO -- would it be faster to just format the data here
        # fully rather than leaving part of it to the data collator?

        return replaced



    def load_model_and_tokenizer_fromdir(checkpoint_directory):

        use_fast=True # Use the fast one!
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_directory, use_fast=use_fast, mlm=True)

        if use_fast:
            print("USING FAST TOKENIZER!")

        if 'mask_token' not in tokenizer.special_tokens_map:
            tokenizer.add_special_tokens({'mask_token':'<mask>'})

        model = AutoModelForMaskedLM.from_pretrained(checkpoint_directory)

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        return tokenizer, model


    def mlm_data_collator(features):
        """
        """
        pad_token = -100
        mask_token_id = tokenizer.mask_token_id

        for feature in features:
            #replaced = feature.pop("replaced")
            replaced = feature['input_ids']
            original = feature['labels']

            # note labels are the original uncorrupted version. Will only compute
            # loss over the modified tokens so:
            newlabels = [-100 if replaced[ii]==original[ii] else i for ii,i in enumerate(original) ]
            feature['labels'] = newlabels
            feature['input_ids'] = [i if replaced[ii] == original[ii] else mask_token_id for ii,i in enumerate(original)]

        return default_data_collator(features)


    def mlm_data_collator_fill(features):
    """ In addition to the targeted MLM, add random masks so that 15% of tokens are masked.
    """
        pad_token = -100
        mask_token_id = tokenizer.mask_token_id

        for feature in features:
            #replaced = feature.pop("replaced")
            replaced = feature['input_ids']
            original = feature['labels']

            # note labels are the original uncorrupted version. Will only compute
            # loss over the modified tokens so:
            newlabels = [-100 if replaced[ii]==original[ii] else i for ii,i in enumerate(original) ]
            #feature['labels'] = newlabels

            newinputs = [i if replaced[ii] == original[ii] else mask_token_id for ii,i in enumerate(original)]

            valid_indices = np.where(np.array(newinputs) != mask_token_id)[0]
            num_replaced = len(newinputs) - len(valid_indices)
            N = max(0, int(len(newinputs)*.15) - num_replaced)
            random_indices = np.random.choice(valid_indices, size=N, replace=False)
            for index in random_indices:
                newlabels[index] = original[index]
                newinputs[index] = mask_token_id

            feature['input_ids'] = newinputs
            feature['labels'] = newlabels

        return default_data_collator(features)
    

    def clm_data_collator(features):
        """
        """
        pad_token = -100
        mask_token_id = tokenizer.mask_token_id

        for feature in features:
            #replaced = feature.pop("replaced")
            replaced = feature['input_ids']
            original = feature['labels']

            # note labels are the original uncorrupted version. Will only compute
            # loss over the modified tokens so:
            newlabels = [-100 if replaced[ii]==original[ii] else i for ii,i in enumerate(original) ]
            feature['labels'] = newlabels

        return default_data_collator(features)

    if data_collator_type == 'clm':
        selected_data_collator = clm_data_collator

    if data_collator_type == 'mlm':
        selected_data_collator = mlm_data_collator

    if data_collator_type == 'mlmfill':
        selected_data_collator = mlm_data_collator_fill
    
    #num_samples = 1000 #None
    num_samples = None

    if from_scratch:
        tokenizer = AutoTokenizer.from_pretrained('tokenizers/10m_maestro/', use_fast=True)
        config_kwargs = {
            "cache_dir": None,
            "revision": "main",
            "use_auth_token": False,#True if model_args.use_auth_token else None,
            "vocab_size": tokenizer.vocab_size,
        }
        print("Going to train model from scratch.")
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-small", **config_kwargs)

        #small = True
        #small = False
        #if small:
        #    config.hidden_size = 256
        #    config.pooler_hidden_size = 256
        #    config.intermediate_size = 1024
        #    config.num_attention_heads = 4

        model = AutoModelForMaskedLM.from_config(config)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer, model = load_model_and_tokenizer_fromdir(load_checkpoint_dir)#
        #"debertav3small-10m-maestro-generator-finetuned/checkpoint-13941")

    num_parameters = model.num_parameters() / 1_000_000
    print(f"'>>> Number of parameters: {round(num_parameters)}M'")


    dataset = load_original_and_replaced(train_dir, num_samples=num_samples)

    tokenized_dataset = dataset.map(partial(tokenize_function2, tokenizer),
                                    batched=True,
                                    remove_columns=["origtext", "repltext"])

    print("Done tokenizing dataset")

    #NOTE for full model size use 236
    #batch_size = 64#230 #400 #64

    # Show the training loss with every epoch
    #print(logging_steps)
    #model_name = model_checkpoint.split("/")[-1]
    #    model_name = "deberta"
    #model_name = 'tmp-experiment-clmlinguistic'#-generator'
    #outputdirectory

    training_args = TrainingArguments(
        output_dir=outputdirectory,
        overwrite_output_dir=True,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=100,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=num_epochs,
        push_to_hub=False,
        # THIS IS IMPORTANT, because need to keep `word_ids` during training
        remove_unused_columns = False,
        fp16=False, #speed boost
        logging_steps=10,
        save_total_limit=5,
        report_to='wandb',
        optim='adamw_torch_fused',
        torch_compile=True,
        load_best_model_at_end=False,
        disable_tqdm=True
    )

    #TODO need custom Trainer with custom loss? Maybe not
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=selected_data_collator,#, whole_word_masking_data_collator,#data_collator,
        tokenizer=tokenizer
        #compute_metrics= #function to compute metrics here
    )

    print("Training!")
    trainer.train()
    trainer.save_model()

    wandb.finish()

if __name__ == "__main__":
    main()

