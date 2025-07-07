import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

import json 
import argparse
import time

def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)

    parser.add_argument("--sampling-alg", type=str, default='conv')

    parser.add_argument("--prefill", action="store_true", help="use dkv-cache-prefill")
    parser.add_argument("--decode", action="store_true", help="use dkv-cache-decode")
    parser.add_argument("--pd", action="store_true", help="use dkv-cache-pd")
    parser.add_argument("--conv", action="store_true", help="use dkv-cache-conv")
    

    parser.add_argument("--cache-steps", type=int, default=4, help="number of steps to cache")
    args = parser.parse_args()

    model_path = "Dream-org/Dream-v0-Instruct-7B"

    if args.decode or args.prefill:
        from models.modeling_dream import DreamModel
        model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    elif args.pd:
        from models.modeling_dream_pd import DreamModel
        model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    elif args.conv:
        from models.modeling_dream_conv import DreamModel
        model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    else:
        from models.modeling_dream_original import DreamModel
        model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to("cuda").eval()

    question_1 = "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
    question_2 = 'Write a story that ends with "Finally, Joey and Rachel get married."'
    question_3 = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"   
    question_4 = "Give me a short introduction to large language model?"
    question_5 = "Can you introduce something about Paris?"
    question_6 = "Write a code for quicksort. "


    messages = [[
        {
            "role": "user", 
            "content": "If Marcy works for the same company for 40 years, she gets an annual pension of $50,000/year. Starting after 20 years, she becomes entitled to 5% of the value of the pension per year. If she quits after 30 years, what will her annual pension be?"
            # "content": "Mishka bought 3 pairs of shorts, 3 pairs of pants, and 3 pairs of shoes. One pair of shorts costs $16.50. One pair of pants costs $22.50 and one pair of shoes costs $42. How many dollars did Mishka spend on all the clothing items?"
        }
    ]
    # ], [
    #     {
    #         "role": "user", 
    #         "content": question_2
    #     }
    # ]
    # ], [
    #     {
    #         "role": "user", 
    #         "content": "Answer the question step by step and put the answer in \\boxed\{\}: " + question_3
    #     }
    # ], [
    #     {
    #         "role": "user",
    #         "content": question_4
    #     }
    # ], [
    #     {
    #         "role": "user",
    #         "content": question_5
    #     }
    # ], [
    #     {
    #         "role": "user",
    #         "content": question_6
    #     }
    # ]
    
    ]
    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = prompt_ids.input_ids.to(device="cuda")
    attention_mask = prompt_ids.attention_mask.to(device="cuda")
    
   
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.seq_len,
        output_history=True,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=0.0,
        top_p=0.95,
        alg=args.sampling_alg,
        alg_temp=0.,
        use_cache=args.decode or args.prefill or args.pd or args.conv,
        cache_type="decoded",
        cache_steps=args.cache_steps,
        shift_type="un"
    )
    
    for b in range(len(messages)):
        print()
        print(f"----Question {b+1}: {messages[b][0]['content']}")
        sequence = output.sequences[b]
        print(tokenizer.decode(sequence[len(input_ids[0]):]).split('<|endoftext|>')[0])
    
        