from transformers import T5TokenizerFast
import argparse

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('--tokenizer_file', type=str, required=True)
    argp.add_argument('--output_dir', type=str, required=True)
    args = argp.parse_args()
    
    tokenizer = T5TokenizerFast(vocab_file=args.tokenizer_file, extra_ids=0)
    tokenizer.mask_token = '<mask>'
    tokenizer.cls_token = '<cls>'
    tokenizer.model_max_length = 512
    tokenizer.save_pretrained(args.output_dir)
