import torch
import numpy as np
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from models.modeling_llada_dkv_cache_decode import LLaDAModelLM

import time

from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from contextlib import contextmanager, nullcontext


memory_allocs = []
memory_peaks = []

def mem(msg=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    memory_allocs.append(alloc)
    memory_peaks.append(peak)
    print(f"{msg:30s}  alloc={alloc:5.2f} GB  peak={peak:5.2f} GB")

@contextmanager
def mem_scope(name):
    torch.cuda.reset_peak_memory_stats()
    start=time.time()
    yield
    mem(f"{name}  ({time.time()-start:4.1f}s)")


def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@ torch.no_grad()
def generate(model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, enable_cache=False, cache_reloading_step=1, **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    B, L = prompt.shape
    # get the intial form, which is a batch of [MASK] tokens, except for the prompt. Which is cloned. 
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

# I am not sure what this is doing, but it is concating a false column at the beginning of the special_index tensor, and remove the last column. 
    special_index = (x == 126347)
    special_index = torch.cat([torch.zeros((B, 1), dtype=torch.bool).to(x.device), special_index], dim=1)[:, :-1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

# the generate code actually allocated the steps evenly aomng blocks. 
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    
    for num_block in range(num_blocks):
        past_qkv = None
        # Curious why we need 2 transfer_indexes. 
        prv_transfer_idx, cur_transfer_index = None, None

        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            with mem_scope(f"step {i}"):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    raise NotImplementedError('cfg_scale > 0 is not supported yet for cache.')
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    # if not refreshing the cache, 
                    if i % cache_reloading_step != 0 and i > 1 and enable_cache:
                        # set to reuse both query and kv cache. (essential for reordered Rope, if not set, use the original whole seq pos emb.)
                        model.set_qkv_cache(True, True)

                        # this sets internally winthin the rope class, the reordered positional embedding cache.
                        model.get_pos_rotary_embedding_cache(~prv_transfer_idx)
                        # next_x is all tokens still masked. 
                        next_x = x[~prv_transfer_idx].view(x.shape[0], -1)
                        outputs = model(next_x, 
                            past_query_key_values = past_qkv, 
                            use_cache = True, preprocess_cache=True
                        )
                        
                        model.set_qkv_cache(False, False)
                        # rest all __pos_cache to None, whenever we reuse the cache. 
                        # the next time we use the cache, regenerate the positional embedding cache. (they are already generated, just need to be reordered. )
                        model.reset_pos_rotary_embedding_cache()
                        
                    # this corresponds to a cache refresh step, where use_cache is set to False, and preprocess_cache is set to True.
                    elif i > 0 and enable_cache:
                        outputs = model(
                            x, past_query_key_values = past_qkv, 
                            use_cache = False, preprocess_cache = True
                        )
                    else:
                        outputs = model(
                            x, past_query_key_values = None, 
                            use_cache = False, preprocess_cache = False
                        )
                        
                    # need to investigate what is past_qkv.
                    logits = outputs.logits
                    past_qkv = outputs.past_key_values

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature) # b, l, D
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1) # B L V
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l, the probability of the predicted tokens. 
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                elif remasking == 'ar':
                    x0_p = torch.zeros_like(x0, dtype=torch.float64, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                # after this if clasue, the x0_p is the confidence/prob, why x0 is the actual tokens. 
                if x0_p.shape[1] < x.shape[1]: # for cache
                    # Refill x0_p with the -np.inf
                    refill_x0_p = torch.full((x.shape[0], x.shape[1]), -np.inf, device=x0_p.device, dtype=x0_p.dtype)
                    reorder_token_idx = torch.nonzero(~prv_transfer_idx, as_tuple=True)[1].view(x0_p.shape[0], -1)
                    refill_x0_p = refill_x0_p.scatter_(1, reorder_token_idx, x0_p)
                    
                    x0_p = refill_x0_p
                    x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                    # Refill x0 with mask_id
                    refill_x0 = torch.full((x.shape[0], x.shape[1]), mask_id, device=x0.device, dtype=x0.dtype)
                    refill_x0 = refill_x0.scatter_(1, reorder_token_idx, x0)
                    x0 = refill_x0
                
            # Keep the predictions from x0 only where the token was masked, 
            # and preserve the original input x elsewhere (like prompt tokens).
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                if remasking == 'ar':
                    masked_pos = torch.nonzero(mask_index[j], as_tuple=True)[0]
                    select_index = masked_pos[:num_transfer_tokens[j, i]]
                else:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index] 
            all_transfer_index = (x != mask_id) 
            fix_remove = False
            if fix_remove:
                all_transfer_index = all_transfer_index & (~special_index)

            # TODO: check if prv-transfer-idx and cur-transfer-idx work at the final order of the sequence
            # So prv_transfer_idx stores a snapshot of which tokens were already generated before this step, 
            # and cur_transfer_index reflects the tokens that are now known after the current step.
            prv_transfer_idx, cur_transfer_index = cur_transfer_index, all_transfer_index
            past_qkv = [past_qkv, (prv_transfer_idx, cur_transfer_index)]

            #print(x[:, prompt.shape[1]:]) # For Debug
            #for b in range(B):

        #print("Block {}: {}".format(num_block, tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]))
        #print(x[:, prompt.shape[1]:])
        
    if len(memory_allocs) > 0:
        print("=" * 60)
        print(f"Average Allocated Memory: {np.mean(memory_allocs):.2f} GB")
        print(f"Average Peak Memory:      {np.mean(memory_peaks):.2f} GB")
        print(f"Max Peak Memory:      {np.max(memory_peaks):.2f} GB")
        
        print("=" * 60)

    return x


def main():
    device = 'cuda:1'

    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = [
        "John plans to sell all his toys and use the money to buy video games. He has 13 lego sets and he sells them for $15 each. He ends up buying 8 video games for $20 each and has $5 left. How many lego sets does he still have?",
    ] * 24

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [[{"role": "user", "content": "Please answer the question step by step and put the answer in \\boxed{}." + p}] for p in prompt]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    print(prompt)

    bsz = len(prompt)

    input_ids = tokenizer(
        prompt,
        padding_side = 'left',
        padding = 'longest'
    )['input_ids']
    input_ids = torch.tensor(input_ids).to(device)#.unsqueeze(0)#.repeat(bsz, 1)

    #set_random_seed(42)
    decoding_start_time = time.time()
    out = generate(
        model, tokenizer, input_ids, 
        steps=128, gen_length=128, block_length=32, 
        temperature=0., cfg_scale=0., 
        remasking='low_confidence',
        enable_cache=True,
        cache_reloading_step=4
    )
    print("Total Decoding Time = ", time.time() - decoding_start_time)
    res = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for r in res:
        print(r, '\n')

    
def run_profile():
    device = 'cuda:7'
    
    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = [
        "John plans to sell all his toys and use the money to buy video games. He has 13 lego sets and he sells them for $15 each. He ends up buying 8 video games for $20 each and has $5 left. How many lego sets does he still have?",
    ] * 16

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [[{"role": "user", "content": "Please answer the question step by step and put the answer in \\boxed{}." + p}] for p in prompt]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    print(prompt)

    bsz = len(prompt)

    input_ids = tokenizer(
        prompt,
        padding_side = 'left',
        padding = 'longest'
    )['input_ids']
    input_ids = torch.tensor(input_ids).to(device)#.unsqueeze(0)#.repeat(bsz, 1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler("./prof_log", worker_name="decode"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        out = generate(
            model, tokenizer, input_ids,
            steps=128, gen_length=128, block_length=32,
            temperature=0., remasking='low_confidence',
            enable_cache=True, cache_reloading_step=4,
        )
        
        torch.cuda.synchronize()
    # quick console summary
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # print(prof.key_averages(group_by_input_shape=True)
    #           .table(sort_by="cuda_time_total", row_limit=20))

    # scopes = [evt for evt in prof.events()
    #         if evt.name in {"cat_past_kv",               # or whatever you used
    #                         "scatter_reorder"}]

    # for s in scopes:
    #     print(f"{s.name:<20}  cpu_time={s.cpu_time_total/1e6:6.2f} ms  "
    #         f"cuda_mem={s.self_cuda_memory_usage/1e6:6.1f} MB  "
    #         f"#calls={s.count}")

    # print(
    #     prof.key_averages()
    #         .table(sort_by="cuda_time_total", row_limit=20))
    
    # Sort by CUDA memory allocation
    print(prof.key_averages()
            .table(sort_by="self_cuda_memory_usage", row_limit=100))

    # Optionally: access memory usage from your label
    # for evt in prof.key_averages():
    #     if "kv" in evt.key:
    #         print(f"[{evt.key}] cuda mem = {evt.self_cuda_memory_usage/1e6:.2f} MB, cpu time = {evt.self_cpu_time_total:.2f} us")

if __name__ == "__main__":
    torch.cuda.set_device(7)        # ← run on a single GPU to keep traces tidy
    run_profile()
    # main()


# if __name__ == '__main__':
#     main()