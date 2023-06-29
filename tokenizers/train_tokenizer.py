import sentencepiece as spm
import argparse

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('input_files', type=str, nargs='+')
    argp.add_argument('-k', '--vocab-size', type=int, required=True)
    argp.add_argument('-o', '--output-prefix', type=str, required=True)
    argp.add_argument('--split-digits', action='store_true')
    argp.add_argument('--byte-fallback', action='store_true')
    args = argp.parse_args()

    spm.SentencePieceTrainer.train(
        input=','.join(args.input_files),
        model_type='unigram',
        model_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        split_digits=args.split_digits,
        byte_fallback=args.byte_fallback,
        max_sentence_length=1000000,
        pad_id=5,
        control_symbols='<mask>,<cls>'
    )
