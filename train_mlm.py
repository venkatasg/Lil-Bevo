# https://huggingface.co/learn/nlp-course/chapter7/3?fw=pt
"""
See modification in whole_word_masking_data_collator which replaces
instead of masking, using another generator model.

i.e., "simplified CLM" from "METRO: Efficient Denoising Pretraining of Large Scale Autoencoding
Language Models with Model Generated Signals"

TODO:
- multitask learning with added RTD loss.

"""
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import default_data_collator
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

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

#NOTE: I may not need this after all, just change the data_collator
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


def load_data(train_dir, dev_dir, num_samples=None):
    """
    Currently just removing very short lines... not sure if I should be throwing
    this away. But it seems odd to just """
    #TODO do grouping

    print("Loading data...")
    with open(os.path.join(dev_dir, "dev.txt"),'r') as fd:
        devtext = fd.readlines()
    devtext = [i.strip() for i in devtext if i.strip()!=""]
    devtext = [i for i in devtext if len(i.strip())>=5 and len(i.strip().split(" "))>4]
    #devtokens = [i.rstrip().split(" ") for i in devtext]

    with open(os.path.join(train_dir, "train.txt"),'r') as fd:
        traintext = fd.readlines()

    traintext = [i.strip() for i in traintext if i.strip()!=""]
    traintext = [i for i in traintext if len(i.strip())>=5 and len(i.strip().split(" "))>4]
#    traintokens = [i.rstrip().split(" ") for i in traintext]

    if num_samples is not None:
        devtext = devtext[:num_samples]
        traintext = traintext[:num_samples]
#        devtokens = devtokens[:num_samples]
#        traintokens = traintokens[:num_samples]

#    devdata = Dataset.from_dict({"text":devtokens})
#    traindata = Dataset.from_dict({"text":traintokens})
    devdata = Dataset.from_dict({"text":devtext})
    traindata = Dataset.from_dict({"text":traintext})


#    devdata = Dataset.from_dict({"tokens":devtokens})
#    traindata = Dataset.from_dict({"tokens":traintokens})

    dataset = DatasetDict({"train": traindata, "validation": devdata} )
    print("Done loading data...")
    return dataset


def tokenize_function(tokenizer, examples):
    """
    Note, this also adds the labels, which are just a copy of the input_ids
    so we have ground truth for the LM.
    """
    max_seq_length = 128 #32

    #TODO 128 is GOOD change it back later128
    #if not tokenizer.is_fast:
    #    raise ValueError("MUST USE A FAST TOKENIZER")
    result = tokenizer(examples["text"],
                       truncation=True,
                       padding='max_length',
                       max_length = max_seq_length,
                    )
    result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

    result["labels"] = result["input_ids"].copy()

    return result


def load_model_and_tokenizer_fromdir(checkpoint_directory):

    use_fast=True # Use the fast one!
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_directory, use_fast=use_fast, mlm=True)

    if use_fast:
        print("USING FAST TOKENIZER!")

    if 'mask_token' not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens({'mask_token':'<mask>'})

    model = AutoModelForMaskedLM.from_pretrained(checkpoint_directory)
    #"checkpoints/deberta16kvocab/deberta/")

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model




#def main():

# This model is pretty good, but there is something that fails when use_fast=True
# Maybe the model is too good. Use the 5th prediction.
model_checkpoint = "checkpoints/deberta-fast/deberta-512/checkpoint-18000/"
genmodel = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
gentokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)


def whole_word_masking_data_collator(features):
    """
    Will use a different masking for every batch!

    ex = tokenized_dataset['train'][10]
    batch = whole_word_masking_data_collator([ex])
    """
    #NOTE only labels for MASKED words are not -100

    wwm_probability= 0.15

    # I think this is right... not just blindly use -100
    #https://github.com/huggingface/transformers/issues/22634
    #https://github.com/huggingface/transformers/pull/18592

    #TODO check
    #pad_token = tokenizer.pad_token_id
    pad_token = -100
    mask_token_id = tokenizer.mask_token_id
    #TODO check -- looks like 'labels' does not keep punctuation token_ids



    # like 1012 (for '.') -- these are overwriten by -100
    # ACTUALLY THIS HAPPENS WHEN USING tokenizer.encode


    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]

        new_labels = [pad_token] * len(labels)
#        replaced = [0]* (len(labels)) # convention: 1 for replaced, 0 else
        #new_labels = [-100] * len(labels)

        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = mask_token_id
                #replaced[idx] = 1
        feature["labels"] = new_labels


        # Predict using auxiliary model
        #input_ids.to(genmodel.device)
        token_logits = genmodel(torch.tensor(input_ids).unsqueeze(0)).logits
        for idx, x in enumerate(input_ids):
            if x == mask_token_id:
                mask_token_logits = token_logits[0, idx, :]
                #print(type(mask_token_logits))
                #print(mask_token_logits.shape)
                #print(mask_token_logits)
                # 5th best option:
                top_n_tokens = torch.topk(mask_token_logits, 5).indices.tolist()
                #print(top_n_tokens)
                #for token in top_n_tokens:
                #    print( gentokenizer.decode([token] ) )
                input_ids[idx] = top_n_tokens[2]  # use 3rd best option

        # Should track replaced tokens!
        #replaced = torch.where(feature['input_ids'] == mask_token_id , 1 , 0)
        #assert torch.equal(replaced,  torch.where( batch['labels'] !=-100 , 1 , 0) )
        #feature["replaced"] = replaced

    return default_data_collator(features)

#num_samples = 100# set to None later to train on all of it.

num_samples = None

dev_dir = 'babylm_data/babylm_dev'
train_dir = 'babylm_data/babylm_10M'


#TODO either load a fast pretrained deberta or load from scratch


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

#model_checkpoint = "distilbert-base-uncased"
#model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#tokenizer, model = load_model_and_tokenizer_fromdir("checkpoints/deberta16kvocab/deberta/")

num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> Number of parameters: {round(num_parameters)}M'")

# Original datacollator -- but we want to do whole word masking!
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

dataset = load_data(train_dir, dev_dir, num_samples=num_samples)

tokenized_dataset = dataset.map(partial(tokenize_function, tokenizer),
                                batched=True,
                                remove_columns=["text"])
print("Done tokenizing dataset")


#NOTE for full model size use 236
batch_size = 230 #400 #64

# Show the training loss with every epoch
logging_steps = len(tokenized_dataset["train"]) // batch_size
#print(logging_steps)
#model_name = model_checkpoint.split("/")[-1]
#    model_name = "deberta"
model_name = 'debertav3small-10m-maestro'#-generator'

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned",
    overwrite_output_dir=True,
    evaluation_strategy="no",#"epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=50,
    push_to_hub=False,
    # THIS IS IMPORTANT, because need to keep `word_ids` during training
    remove_unused_columns = False,
    fp16=True, #speed boost
    logging_steps=logging_steps,
    load_best_model_at_end=False
)

#TODO need custom Trainer with custom loss? Maybe not
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=whole_word_masking_data_collator,#data_collator,
    tokenizer=tokenizer
    #compute_metrics= #function to compute metrics here
)

print("Training!")
trainer.train()
trainer.save_model()

wandb.finish()

#if __name__ == "__main__":
#    main()

