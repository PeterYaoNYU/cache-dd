import torch
import numpy as np
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from models.modeling_llada_cache_conv import LLaDAModelLM

import time


import re
_NUMBER_RE = re.compile(r'\d[\d,]*(\.\d+)?')  
def id_is_numeric(tok_id, tokenizer) -> bool:
    """
    Return True if `tok_id` corresponds to something that looks like
    a number token – e.g. '7', '123', '3.14', '75,000', etc.
    """
    tok = tokenizer.convert_ids_to_tokens(int(tok_id), skip_special_tokens=True)

    # Many tokenizers put a leading space before regular words,
    # so strip it off first:
    tok = tok.lstrip()
    if bool(re.fullmatch(r'\d[\d,]*(\.\d+)?', tok)):
        print(f"Token {tok} is numeric.")
    # Accept integers, decimals, or comma-separated thousands
    return bool(re.fullmatch(r'\d[\d,]*(\.\d+)?', tok))


# given a sequence, return the idx of number tokens: Only work with batch size = 1 for now., 
def get_number_token_indices(seq: torch.Tensor, tokenizer, start_idx=0) -> torch.Tensor:
    # Convert only the tail we’re interested in (faster):
    tail_ids = seq[start_idx:].tolist()
    tail_toks = tokenizer.convert_ids_to_tokens(tail_ids, skip_special_tokens=True)

    numeric_pos = [
        i + start_idx
        for i, tok in enumerate(tail_toks)
        if _NUMBER_RE.fullmatch(tok.lstrip())
    ]
    
    # for i, tok in enumerate(tail_toks):
    #     if _NUMBER_RE.fullmatch(tok.lstrip()):
    #         print(f"Token {tok} at position {i + start_idx} is numeric.")

    return torch.tensor(numeric_pos, dtype=torch.long, device=seq.device)


def block_starts(x: torch.tensor, n_parallel:int, start: int = 0, num_blocks: int = 1, block_id: int = 0):
    B = x.shape[0]
    L = x.shape[1]
    
    block_len = (L - start) // n_parallel
    
    # storing the index of the start of each block in a tensor.
    starts = torch.empty((B, n_parallel), dtype=torch.long, device=x.device)
    
    # remaining = L - start
    base, rem = divmod(block_len, n_parallel)

    sizes = torch.tensor(
        [base+1] * rem + [base] * (n_parallel - rem),
        device=x.device, dtype=torch.long, 
    )
        
    starts_row = torch.cat((
        torch.zeros(1, device=x.device, dtype=torch.long),
        torch.cumsum(sizes, 0)
    ))[:-1]  
    
    starts = starts_row.expand(B, -1).clone()
    starts += start
    
    starts = starts + block_id * block_len
    
    print("Block starts, ", block_starts)
    return starts
        
        
        
def block_far_mask(block_start_idx: torch.Tensor, L: int, radius: int = 16) -> torch.BoolTensor:
    """
    Parameters
    ----------
    block_start_idx : (B, num_blocks)
        Start indices of blocks per sample.
    L : int
        Sequence length.
    radius : int
        Distance threshold from block start positions.

    Returns
    -------
    mask : (B, L) BoolTensor
        True where token is NOT within `radius` of any block start.
    """
    B, num_blocks = block_start_idx.shape

    # shape: (1, L)
    positions = torch.arange(L, device=block_start_idx.device).unsqueeze(0)

    # shape: (B, L, num_blocks) ← broadcast subtraction
    dist = (positions.unsqueeze(-1) - block_start_idx.unsqueeze(1)).abs()

    # shape: (B, L): is there *any* start within radius?
    is_near_any_block = (dist < radius).any(dim=-1)

    # invert to get final mask: True if it's *not* near any block start
    mask = ~is_near_any_block
    return mask
    
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

from typing import List, Tuple, Optional

def _merge_one_layer(
    k: torch.Tensor,               # (B, N_prev, D) – keys stored *right now*
    v: torch.Tensor,               # (B, N_prev, D) – values stored *right now*
    prev_idx : torch.BoolTensor,   # (B, L) – mask that produced those tensors
    new_idx  : torch.BoolTensor,   # (B, L) – mask we want for the upcoming pass
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep only those rows whose token-positions remain True in `new_idx`.
    No new rows are added here – they will be produced by the model soon.
    """
    B, _, D = k.shape
    k_kept, v_kept = [], []
    for b in range(B):
        # Get the indices of the tokens that are still masked in the new_idx
        prev_masked_indices = torch.nonzero(prev_idx[b], as_tuple=True)[0]
        new_masked_indices = torch.nonzero(new_idx[b], as_tuple=True)[0]

        # get the number of tokens to reduce in the new_masked_indices
        num_to_reduce =  len(prev_masked_indices) - len(new_masked_indices) 
        # print("prev_masked_indices len: ", len(prev_masked_indices), " new_masked_indices len: ", len(new_masked_indices), " num_to_reduce: ", num_to_reduce)
        if num_to_reduce > 0:
            # reduce the number of tokens in k, and v and return. 
            # print("reducing tokens in k and v for batch ", b, " by ", num_to_reduce)
            k_kept.append(k[b, num_to_reduce:, :])
            v_kept.append(v[b, num_to_reduce:, :])
        else:
            # keep all tokens in k and v, and return. 
            k_kept.append(k[b, :, :])
            v_kept.append(v[b, :, :])
    # Concatenate the kept keys and values across all batches
    k = torch.stack(k_kept, dim=0)  # (B, N_new, D)
    v = torch.stack(v_kept, dim=0)  # (B, N_new, D)
    return k, v



def trim_and_update_future_cache(
    future_cache     : Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    future_cache_idx : torch.BoolTensor,   # ← **NEW** mask we want to keep
    prev_cache_idx   : Optional[torch.BoolTensor],  # ← mask that matches cache
) -> Tuple[Optional[List[Tuple[torch.Tensor, torch.Tensor]]], torch.BoolTensor]:
    pruned_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for (k, v) in future_cache:
        k_new, v_new = _merge_one_layer(k, v, prev_cache_idx, future_cache_idx)
        pruned_cache.append((k_new, v_new))
    # print("pruned_cache shape: ", [c[0].shape for c in pruned_cache])
    return pruned_cache, future_cache_idx.clone()

@ torch.no_grad()
def generate(model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, enable_cache=False, cache_reloading_step=4, **kwargs):
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
    threshod = 0.8
    
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
    # todo watch out for corner cases where there is a remainder.
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    print("Number of blocks: ", num_blocks, " Steps per block: ", steps)
    
    print("Prompt length: ", prompt.shape[1], " Gen length: ", gen_length)
    
    # decode_idx = block_starts(x, 2, L)
    
    future_cache = None
    
    for num_block in range(num_blocks):
        past_qkv = None
        # Curious why we need 2 transfer_indexes. 
        prv_transfer_idx, cur_transfer_index = None, None

        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        print("cache refresh step: ", cache_reloading_step, " remasking strategy: ", remasking )
        
        future_cache_idx = None
        
        decode_idx = block_starts(x, 2, num_blocks=num_blocks, block_id=num_block) # (B, 2)
        
        spec_index_to_verify = None
        
        for i in range(steps):
            print("Step: ", i, " decode idx: ", decode_idx)
            mask_index = (x == mask_id)
            if cur_transfer_index is not None:
            # still_masked = (input_ids == 126336) # [Mask] id
                far_enough = torch.cumsum(~cur_transfer_index, dim=1) > 32
                future_cache_idx = far_enough
            if cfg_scale > 0.:
                raise NotImplementedError('cfg_scale > 0 is not supported yet for cache.')
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                
                if i % 32 == 0:
                    print ("Number Correction Step....")
                    numerical_tokens_idx = get_number_token_indices(x[0], tokenizer, L)
                    print("Number tokens idx: ", numerical_tokens_idx.shape)
                    if numerical_tokens_idx.shape[0] > 0:
                        math_correction_iter_count = numerical_tokens_idx.shape[0]
                        # mask the sequence where there are numbers:
                        
                        for b in range(B):
                            x[b, numerical_tokens_idx] = mask_id
                            
                        for j in range(math_correction_iter_count):
                            math_decode_idx = numerical_tokens_idx[j]
                            print("Decode idx: ", math_decode_idx, " math correction decode step: ", j, " token id: ", x[0, math_decode_idx])
                            
                            outputs = model(
                                x, past_query_key_values = None, 
                                use_cache = False, preprocess_cache = False,
                                # future_cache_idx = future_cache_idx,
                            )
                            
                            logits = outputs.logits
                            
                            logits_with_noise = add_gumbel_noise(logits, temperature=temperature) # b, l, D
                            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                            
                            print("Change from x to x0: ", x[0, math_decode_idx], " -> ", x0[0, math_decode_idx])
                            x[b, math_decode_idx] = x0[b, math_decode_idx]
                        continue
                    else:
                        continue

                    
                    
                # print("Decode idx: ", decode_idx," decode step: ", i)
                cache_reuse_mask = block_far_mask(decode_idx, x.shape[1], radius=32)       
                # cache_reuse_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                # if not refreshing the cache, reuse the cache. 
                if i % cache_reloading_step != 0 and i > 1 and enable_cache:
                    # set to reuse both query and kv cache. (essential for reordered Rope, if not set, use the original whole seq pos emb.)
                    if hasattr(model, "module"):
                        model.module.set_qkv_cache(True, True)
                    else:
                        model.set_qkv_cache(True, True)
                    # print("step: ", i, " cache refresh, x shape: ", x.shape)
                    outputs = model(x, 
                        past_query_key_values = past_qkv, 
                        use_cache = True, preprocess_cache=True, 
                        decode_now_idx = cur_transfer_index,
                        future_cache_idx = future_cache_idx, 
                        future_cache = future_cache, 
                        idx_to_decode = decode_idx, 
                        cache_reuse_mask = cache_reuse_mask,
                        # spec_index_to_verify = spec_index_to_verify,
                    )
                    
                    if hasattr(model, "module"):
                        model.module.set_qkv_cache(False, False)
                    else:
                        model.set_qkv_cache(False, False)
                    
                # this corresponds to a cache refresh step, where use_cache is set to False, and preprocess_cache is set to True.
                elif i > 0 and enable_cache:
                    # print("step: ", i, " cache refresh, x shape: ", x.shape)
                    outputs = model(
                        x, past_query_key_values = past_qkv, 
                        use_cache = False, preprocess_cache = True, 
                        future_cache_idx = future_cache_idx,
                        future_cache = future_cache, 
                        idx_to_decode = decode_idx, 
                        cache_reuse_mask = cache_reuse_mask,
                        # spec_index_to_verify = spec_index_to_verify,
                    )
                    
                    # decode_idx = decode_idx + 1
                else:
                    outputs = model(
                        x, past_query_key_values = None, 
                        use_cache = False, preprocess_cache = False,
                        # future_cache_idx = future_cache_idx,
                    )
                    
                # need to investigate what is past_qkv.
                logits = outputs.logits
                past_qkv = outputs.attn_key_values
                future_cache = outputs.future_cache
                # print("step: ", i, "future_cache shape: ", len(future_cache) if future_cache is not None else None)
                prev_cache_idx = future_cache_idx
                
                # print("step: ", i, "pask _qkv shape", past_qkv[0][0].shape if past_qkv[0] is not None else None)

            # print("Step: ", i, "logits shape: ", logits.shape, " decode idx: ", decode_idx)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature) # b, l, D
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            x0 = torch.where(mask_index, x0, x)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1) # B L V
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l, the probability of the predicted tokens. 
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'ar' or 'convolution':
                p = F.softmax(logits.to(torch.float64), dim=-1) # B L V
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
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
            # x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # if spec_index_to_verify is not None:
            #     x0_spec_verify = x0[spec_index_to_verify]
            #     confidence_spec_verify = x0_p[spec_index_to_verify]
            #     # print("x0 spec verify: ", x0_spec_verify, " x0 spec verify shape: ", x0_spec_verify.shape)
            #     print("Step: ", i-1 , "confidence spec verify: ", confidence_spec_verify, " confidence spec verify shape: ", confidence_spec_verify.shape)
            #     bad_token_idx = confidence_spec_verify < threshod

            #     print("Step: ", i, " bad token idx: ", bad_token_idx)
            #     # print("Low confidence mask: ", low_conf_mask)
                

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            select_index = decode_idx.clone().to(x0.device)
            
            for j in range(confidence.shape[0]):
                if remasking == 'ar':
                    masked_pos = torch.nonzero(mask_index[j], as_tuple=True)[0]
                    select_index = masked_pos[:num_transfer_tokens[j, i]]
                elif remasking == 'convolution':
                    select_index.clamp_(max = x.size(1) - 1)
                else:
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                # print("Step: ", i , "select idx confidence: ", confidence[j, select_index], " decode idx: ", decode_idx[j])
            
            decode_idx = decode_idx + 1
            
            
            
            x[transfer_index] = x0[transfer_index] 
            all_transfer_index = (x != mask_id) 
            fix_remove = False
            if fix_remove:
                all_transfer_index = all_transfer_index & (~special_index)
                
            # spec_index_to_verify = transfer_index.clone()

            # TODO: check if prv-transfer-idx and cur-transfer-idx work at the final order of the sequence
            # So prv_transfer_idx stores a snapshot of which tokens were already generated before this step, 
            # and cur_transfer_index reflects the tokens that are now known after the current step.
            prv_transfer_idx, cur_transfer_index = cur_transfer_index, all_transfer_index
            past_qkv = [past_qkv, (prv_transfer_idx, cur_transfer_index)]

            #print(x[:, prompt.shape[1]:]) # For Debug
            #for b in range(B):

        #print("Block {}: {}".format(num_block, tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)[0]))
        #print(x[:, prompt.shape[1]:])

    return x


def main():
    device = 'cuda:0'

    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = [
        "John plans to sell some of his toys and use the money to buy video games. He has 13 lego sets and he sells them for $15 each. He ends up buying 8 video games for $20 each and has $5 left. How many lego sets does he still have after selling and buying?",
    ]

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
        steps=128, gen_length=256, block_length=128, 
        temperature=0., cfg_scale=0., 
        remasking='convolution',
        enable_cache=True,
        cache_reloading_step=4
    )
    print("Total Decoding Time = ", time.time() - decoding_start_time)
    res = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for r in res:
        print(r, '\n')

    


if __name__ == '__main__':
    main()