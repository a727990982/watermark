import os, sys, argparse, time

import numpy as np
from transformers import AutoTokenizer
from mersenne import mersenne_rng

# 先导入和设置 pyximport
import pyximport
pyximport.install(reload_support=True, 
                 language_level=sys.version_info[0],
                 setup_args={'include_dirs': np.get_include()})

# 然后使用完整的相对路径导入 levenshtein
from levenshtein import levenshtein

def permutation_test(tokens,key,n,k,vocab_size,n_runs=100):
    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n*vocab_size)], dtype=np.float32).reshape(n,vocab_size)
    test_result = detect(tokens,n,k,xi)

    p_val = 0
    for run in range(n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
        null_result = detect(tokens,n,k,xi_alternative)

        # assuming lower test values indicate presence of watermark
        p_val += null_result <= test_result

    return (p_val+1.0)/(n_runs+1.0)


def detect(tokens, n, k, xi, gamma=0.0):
    m = len(tokens)
    n = len(xi)
    
    A = np.empty((m-(k-1), n))
    for i in range(m-(k-1)):
        for j in range(n):
            token_slice = tokens[i:i+k].astype(np.int64)
            xi_slice = xi[j].reshape(1,-1)
            A[i][j] = levenshtein(token_slice, xi_slice, gamma)
    
    return np.min(A)


def main(args):
    with open(args.document, 'r') as f:
        text = f.read()
    if args.tokenizer == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
    
    t0 = time.time()
    pval = permutation_test(tokens,args.key,args.n,len(tokens),len(tokenizer))
    print('p-value: ', pval)
    print(f'(elapsed time: {time.time()-t0}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for a watermark in a text document')
    parser.add_argument('document',type=str, help='a file containing the document to test')
    parser.add_argument('--tokenizer',default='facebook/opt-1.3b',type=str,
            help='a HuggingFace model id of the tokenizer used by the watermarked model')
    parser.add_argument('--tokenizer_path',default="/home/huang/LLaMA-Factory/Meta-Llama-3-8B-Instruct",type=str,
            help='the path to the tokenizer to use when using --model_type=llama')
    parser.add_argument('--n',default=256,type=int,
            help='the length of the watermark sequence')
    parser.add_argument('--key',default=42,type=int,
            help='the seed for the watermark sequence')

    main(parser.parse_args())