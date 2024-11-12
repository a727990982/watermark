import os, argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from mersenne import mersenne_rng

def generate_shift(model,prompt,vocab_size,n,m,key,tokenizer):
    rng = mersenne_rng(key)
    xi = torch.tensor([rng.rand() for _ in range(n*vocab_size)]).view(n,vocab_size)
    shift = torch.randint(n, (1,))

    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1, :vocab_size], dim=-1).cpu()
        token = exp_sampling(probs,xi[(shift+i)%n,:]).to(model.device)
        
        # 解码当前token看看是否包含换行符
        token_text = tokenizer.decode([token.item()])
        if '\n' in token_text:
            break
            
        inputs = torch.cat([inputs, token], dim=-1)
        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()

def exp_sampling(probs,u):
    return torch.argmax(u ** (1/probs),axis=1).unsqueeze(-1)

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.model == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)              
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

#     formatted_prompt = f"""Below is a sentence. Your task is to rephrase it in a different way while keeping the exact same meaning. 
# Original sentence: "{args.prompt}" \n
# Rephrased sentence:"""

    formatted_prompt = f"""Below is a sentence. Your task is to extend it to 20 tokens in one sentence while keeping the exact same meaning:
Original sentence: "{args.prompt}" \n
Rephrased sentence:"""

    tokens = tokenizer.encode(formatted_prompt, return_tensors='pt', truncation=True, max_length=2048)

    watermarked_tokens = generate_shift(model,tokens,len(tokenizer),args.n,args.m,args.key, tokenizer)[0][len(tokens[0]):]
    watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

    print(watermarked_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate text watermarked with a key')
    parser.add_argument('--model',default='facebook/opt-1.3b',type=str,
            help='a HuggingFace model id of the model to generate from')
    parser.add_argument('--prompt',default="set input parameters for elegant tracking, available keys: 'ltefile', 'elefile'.",type=str,
            help='an optional prompt for generation')
    parser.add_argument('--m',default=50,type=int,
            help='the requested length of the generated text')
    parser.add_argument('--n',default=5,type=int,
            help='the length of the watermark sequence')
    parser.add_argument('--key',default=42,type=int,
            help='a key for generating the random watermark sequence')
    parser.add_argument('--seed',default=0,type=int,
            help='a seed for reproducibile randomness')
    # parser.add_argument('--model_type',default='opt',type=str,
    #         help='the type of model to use')
    parser.add_argument('--model_path',default="/home/huang/LLaMA-Factory/Meta-Llama-3-8B-Instruct",type=str,
            help='the path to the model to use when using --model_type=llama')

    main(parser.parse_args())
