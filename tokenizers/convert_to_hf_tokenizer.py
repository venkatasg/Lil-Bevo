from transformers import T5Tokenizer
import argparse

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('--tokenizer_file', type=str, required=True)
    argp.add_argument('--output_dir', type=str, required=True)
    args = argp.parse_args()
    
    tokenizer = T5Tokenizer(args.tokenizer_file)
    tokenizer.mask_token = '<mask>'
    tokenizer.cls_token = '<cls>'
    tokenizer.save_pretrained(args.output_dir)
