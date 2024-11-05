import os, argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from mersenne import mersenne_rng

def generate_shift(model,prompt,vocab_size,n,m,key):
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
        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()

def exp_sampling(probs,u):
    return torch.argmax(u ** (1/probs),axis=1).unsqueeze(-1)

def main(args):
    if args.model_type == 'llama' and args.model_path is None:
        raise ValueError("--model_path must be provided when using --model_type=llama")
        
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    # 修改 prompt 格式以适应 Llama 3
    formatted_prompt = f"""Please rephrase this sentence while keeping the meaning unchanged:
{args.prompt} """

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.m,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 解码时跳过输入的 prompt 部分
    prompt_length = len(inputs.input_ids[0])
    generated_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    print(generated_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate text watermarked with a key')
    parser.add_argument('--model',default='facebook/opt-1.3b',type=str,
            help='a HuggingFace model id of the model to generate from')
    parser.add_argument('--prompt',default='The function is to implement a shortest path algorithm.',type=str,
            help='the user prompt to be retold')
    parser.add_argument('--m',default=80,type=int,
            help='the requested length of the generated text')
    parser.add_argument('--n',default=256,type=int,
            help='the length of the watermark sequence')
    parser.add_argument('--key',default=42,type=int,
            help='a key for generating the random watermark sequence')
    parser.add_argument('--seed',default=0,type=int,
            help='a seed for reproducibile randomness')
    parser.add_argument('--model_type', default='opt', 
                       choices=['opt', 'gpt2', 'llama'],
                       help='Type of model to use')
    parser.add_argument('--model_path', default=None, type=str,
                       help='Path to local model weights')

    main(parser.parse_args())
