# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

NUM_BLOCKS = 1
RADIUS = 16

THRESHOLD = 0.5

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

from .cache_utils import PrefillCache, DynamicCache, ConvCache

logger = logging.get_logger(__name__)

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


def block_starts(x: torch.tensor, num_blocks:int, start: int = 0):
    B = x.shape[0]
    L = x.shape[1]
    
    # storing the index of the start of each block in a tensor.
    starts = torch.empty((B, num_blocks), dtype=torch.long, device=x.device)
    
    remaining = L - start
    base, rem = divmod(remaining, num_blocks)

    sizes = torch.tensor(
        [base+1] * rem + [base] * (num_blocks - rem),
        device=x.device, dtype=torch.long, 
    )
    
    print("Block sizes:", sizes, "num_blocks:", num_blocks, "start:", start)
        
    starts_row = torch.cat((
        torch.zeros(1, device=x.device, dtype=torch.long),
        torch.cumsum(sizes, 0)
    ))[:-1]  
    
    starts = starts_row.expand(B, -1).clone()
    starts += start
    
    # print("Block starts, ", block_starts)
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
    
    # print("Block starts:", block_start_idx, "dist: ", dist, " is near any block:", is_near_any_block)
    

    # invert to get final mask: True if it's *not* near any block start
    mask = ~is_near_any_block
    return mask


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    cache_ratio: Optional[float] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        self.use_cache = kwargs.pop("use_cache", False)
        self.cache_type = kwargs.pop("cache_type", None)
        self.cache_steps = kwargs.pop("cache_steps", 0)
        self.shift_type = kwargs.pop("shift_type", None)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)
        
        
        # print("Use cache:", self.use_cache, "cache type:", self.cache_type, "cache steps:", self.cache_steps)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            print("generation config is none, Using model generation config")
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True
            
        use_cache = kwargs.pop("use_cache", None)
        if use_cache is not None:
            generation_config.use_cache = use_cache
        cache_type = kwargs.pop("cache_type", None)
        if cache_type is not None:
            generation_config.cache_type = cache_type
        cache_steps = kwargs.pop("cache_steps", None)
        if cache_steps is not None:
            generation_config.cache_steps = cache_steps
            
        shift_type = kwargs.pop("shift_type", None)
        if shift_type is not None:
            generation_config.shift_type = shift_type
            
            
        alg = kwargs.pop("alg", None)
        if alg is not None:
            generation_config.alg = alg
            
            
        print(f"prepare_config: Using cache: {generation_config.use_cache}, cache type: {generation_config.cache_type}, cache steps: {generation_config.cache_steps}")

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        
        print("Input id shape:", input_ids.shape)
        
        print("mask token id ", generation_config.mask_token_id, " eos token id: ", generation_config.eos_token_id, " pad token id: ", generation_config.pad_token_id)
        print("bos token id ", generation_config.bos_token_id,)
        

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        **model_kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        
        print("Sample_: use_cache:", generation_config.use_cache,
              "cache_type:", generation_config.cache_type,
              "cache_steps:", generation_config.cache_steps,)

        histories = [] if (return_dict_in_generate and output_history) else None
        
        prompt_length = input_ids.shape[1]

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        # print("x shape:", x.shape, "max_length:", max_length, "max_new_tokens:", generation_config.max_new_tokens)
        B, L = x.shape
        prompt_mask = (x != mask_token_id)
        #print("x shape:", x.shape, generation_config.max_length, generation_config.max_new_tokens)

        # Setting the cache position for prefill. Would need to be changed if use dynamic cache for decoding.
        if generation_config.use_cache:
            print("Using cache....")
            if generation_config.cache_type == "prefill":
                past_key_values = PrefillCache(self.config.num_hidden_layers)

                cache_position = ~(x == mask_token_id)
                cache_position = torch.cat([cache_position[:, 1:], cache_position[:, -1:]], dim=-1)
            
                prv_cache_position = None
            elif generation_config.cache_type == 'decoded' or generation_config.cache_type == 'conv':
                past_key_values = ConvCache(self.config.num_hidden_layers)
                cache_position = ~(x == mask_token_id)
                cache_position = torch.cat([cache_position[:, 1:], cache_position[:, -1:]], dim=-1)
                prv_cache_position = None
            else:
                print(f"Using cache type: {generation_config.cache_type}")
                raise NotImplementedError(f"Unknown cache type: {generation_config.cache_type}")
        else:
            past_key_values = None
            cache_position = None
            prv_cache_position = None
       
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            print("attention mask is NOT none")
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            ori_attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            print("attention mask is none")
            
            tok_idx = torch.arange(
                max_length, device=input_ids.device
            ).unsqueeze(0).repeat(input_ids.shape[0], 1)
            #attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            attention_mask = "full"
            ori_attention_mask = "full"
            #print("In Full:", attention_mask)

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        
        print("prompt len: ", input_ids.shape[1], "max_length:", max_length, )
        
        decode_idx = block_starts(x, num_blocks=NUM_BLOCKS, start=input_ids.shape[1])
        print("decode_idx:", decode_idx)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        
        correction_flag = False
        
        total_tokens, run_tokens = 0, 0
        for i in range(steps//NUM_BLOCKS):
            # print(f"Step {i} of {steps}, decode_idx: {decode_idx}")
            mask_index = (x == mask_token_id)
            
            cache_reuse_mask = block_far_mask(decode_idx, x.shape[1], radius=RADIUS)
            # print("cache_reuse_mask:", cache_reuse_mask)
            # print("decode_idx:", decode_idx)


            if i % generation_config.cache_steps == 0:
                print("refreshing cache at step", i)
                prv_cache_position = None
                past_key_values.refresh_cache()

            total_tokens += (x.shape[1] * x.shape[0])
            if generation_config.use_cache and prv_cache_position is not None:
                step_x = x

                # reorder attention mask
                if attention_mask != "full":
                    print("WARNING: Reordering attention mask")
                    att_mask_reorder = torch.cat([
                        attention_mask[~prv_cache_position].view(x.shape[0], -1),
                        attention_mask[prv_cache_position].view(x.shape[0], -1),
                    ], -1)
                    cur_attention_mask = F.pad(att_mask_reorder, (0, max_length - att_mask_reorder.shape[1]), value=1.0)
            
                    # attention_mask is of shape [B, N]
                    # broadcast to [B, 1, N, N]
                    cur_attention_mask = torch.logical_and(
                        cur_attention_mask.unsqueeze(1).unsqueeze(-2),
                        cur_attention_mask.unsqueeze(1).unsqueeze(-1),
                    )
                else:
                    cur_attention_mask = "full"
            else:
                step_x = x
                cur_attention_mask = ori_attention_mask
            run_tokens += (step_x.shape[1] * step_x.shape[0])

            logits = self(
                input_ids=step_x, 
                attention_mask=cur_attention_mask,
                position_ids=tok_idx,
                use_cache=generation_config.use_cache,
                cache_position=cache_position,
                prv_cache_position=prv_cache_position,
                past_key_values=past_key_values, 
                cache_reuse_mask=cache_reuse_mask,
            ).logits

            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)
            
            correction_flag = False
            correction_indices = []
            
            
            # if i > 0 and i % generation_config.cache_steps == 0:
            #     # speculative verification
            #     logits_with_noise = add_gumbel_noise(logits, temperature)
            #     x0_verify = torch.argmax(logits_with_noise, dim=-1)
            #     x0_verify = x0_verify[~mask_index]
            #     p = F.softmax(logits.to(torch.float64), dim=-1)
            #     x0_p_verify = torch.squeeze(
            #         torch.gather(p, dim=-1, index=torch.unsqueeze(x, -1)), -1)[~mask_index][prompt_length:]
            #     # print(f"Step {i} verify_token: ", x0_verify)
            #     fail_mask = x0_p_verify < THRESHOLD
            #     fail_indices = fail_mask.nonzero(as_tuple=True)[0]
            #     fail_indices = fail_indices + prompt_length
            #     # print(f"Step {i} fail indices: ", fail_indices)
                
            #     fail_idx = torch.zeros_like(x, device=self.device, dtype=torch.bool)
            #     for j in range(x.shape[0]):
            #             fail_idx[j, fail_indices] = True
                
            #     x0_p, x0 = sample_tokens(logits[fail_idx], temperature=temperature, top_p=top_p, top_k=top_k)
                
            #     # print("Shape of x0_p: ", x0_p.shape, "Shape of x0: ", x0.shape, " Shape of x0_verify: ", x0_verify.shape) 
            #     # print("fail indices: ", fail_indices.shape)
            #     for idx in range(fail_indices.shape[0]):
            #         # print("Original token: ", x[0][idx], "prob: ", x0_p[idx], " new token: ", x0[idx])
            #         if x[0][idx] != x0[idx]:
            #             print("correcting token: ",  fail_indices[idx], ", token id ", x[0][idx], " with new token: ", x0[idx])
            #             x[0][idx] = x0[idx]
                
                
            
            
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            
            # print("decoding alg is: ", alg)
            
            
            transfer_idx = torch.zeros_like(x, device=self.device, dtype=torch.bool)
            # if not correction_flag:
            for j in range(x.shape[0]):
                    transfer_idx[j, decode_idx] = True
            # elif correction_flag and len(correction_indices) > 0:
            #     for j in range(x.shape[0]):
            #             transfer_idx[j, correction_indices] = True
                        
                
            # print(f"Step {i} transfer_idx: {transfer_idx}")
            
            if alg == 'conv':
                # print("logits[transfer_idx]:", logits[transfer_idx])
                x0_p, x0 = sample_tokens(logits[transfer_idx], temperature=temperature, top_p=top_p, top_k=top_k)  
                # print(f"Step {i} sample tokens: x0_p: {x0_p}, x0: {x0}")
                # print("done sample tokens, x0_p:", x0_p, "x0:", x0)
                x[transfer_idx] = x0.clone()
                # print(f"Step {i} conv: x: {x},\n x0: {x0}")
            elif alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
                full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                    else:
                        confidence = confidence / alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                    x[row_indices,transfer_index] = x_[row_indices,transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            #print(f"Step {i}:", x)
            if histories is not None:
                histories.append(x.clone())
            
            if not correction_flag:
                decode_idx = decode_idx + 1
                

            if generation_config.use_cache:

                if generation_config.cache_type == "prefill":
                    #print("In prefill")
                    if prv_cache_position is None:
                        prv_cache_position = cache_position
                        cache_position = None  
                elif generation_config.cache_type == "decoded" or generation_config.cache_type == "conv":
                    prv_cache_position = cache_position
                    #print(generation_config.cache_type, generation_config.shift_type, generation_config.cache_steps)
                    if generation_config.shift_type == "un":
                        cache_position = (x != mask_token_id)
                        cache_position = torch.cat([cache_position[:, 1:], cache_position[:, :1]], dim=-1)
                    elif generation_config.shift_type == "un_right":
                        cache_position = (x[:, :-1] != mask_token_id) & (x[:, 1:] != mask_token_id)
                        cache_position = torch.cat([cache_position, cache_position[:, -1:]], dim=-1)
                    elif generation_config.shift_type == "right":
                        cache_position = (x != mask_token_id)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError(f"Unknown cache type: {generation_config.cache_type}")

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
                cache_ratio= run_tokens / total_tokens,
            )
        else:
            return x