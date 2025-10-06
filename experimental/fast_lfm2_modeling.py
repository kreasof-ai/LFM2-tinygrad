"""
LFM2 (Liquid Foundation Model 2) tinygrad Implementation

This version is adapted to load pretrained weights from Hugging Face Hub.
It includes fixes for stateful convolution caching during generation and uses Conv1d.

Heavily inspired from https://github.com/kyegomez/LFM2 and official https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py implementation
"""


import json
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union
# Third-party imports
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch # only used for converting weights
from tqdm import trange
from transformers import AutoTokenizer

# tinygrad imports
from tinygrad import Tensor, dtypes, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.nn import Conv1d, Embedding, Linear, RMSNorm

from paged_attention import PagedKVCache, PageTable

# --- HF-like Model Output ---

@dataclass
class CausalLMOutputWithPast:
    """
    Base class for causal language model (or autoregressive) outputs.
    Adapted for tinygrad from Hugging Face's CausalLMOutputWithPast.
    """
    logits: Tensor
    past_key_values: Optional[List[Any]] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None

# --- Configuration ---

@dataclass
class LFM2Config:
    """Configuration class for LFM2, adapted for Hugging Face models."""
    vocab_size: int = 65536
    hidden_size: int = 1024
    intermediate_size: int = 6656
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    full_attn_idxs: List[int] = field(default_factory=lambda: [2, 5, 8, 10, 12, 14])
    conv_kernel_size: int = 3
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # New fields for Paged Attention
    page_size: int = 16
    max_batch_size: int = 8
    num_pages: int = 2048 # (num_pages * page_size) should be > max_context_length

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "LFM2Config":
        """Creates a config instance from a Hugging Face config dictionary."""
        intermediate_size = config_dict.get("block_ff_dim", config_dict.get("intermediate_size"))

        if config_dict.get("block_auto_adjust_ff_dim", False):
            intermediate_size = int(2 * intermediate_size / 3)
            multiple_of = config_dict.get("block_multiple_of", 256)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            intermediate_size=intermediate_size,
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict.get("num_attention_heads", config_dict.get("num_heads")),
            num_key_value_heads=config_dict["num_key_value_heads"],
            full_attn_idxs=config_dict["full_attn_idxs"],
            conv_kernel_size=config_dict.get("conv_L_cache", 3),
            max_position_embeddings=config_dict["max_position_embeddings"],
            rms_norm_eps=config_dict.get("norm_eps", config_dict.get("block_norm_eps")),
            rope_theta=config_dict["rope_theta"],
        )

def _precompute_rope_cache(dim: int, max_seq_len: int, base: float) -> Tuple[Tensor, Tensor]:
    """Pre-computes the rotary positional embeddings for the given dimensions."""
    inv_freq = 1.0 / (base ** (Tensor.arange(0, dim, 2, dtype=dtypes.float32) / dim))
    t = Tensor.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = t.reshape(-1, 1) * inv_freq.reshape(1, -1)
    emb = Tensor.cat(freqs, freqs, dim=-1)
    # Return as two separate tensors for cos and sin
    return emb.cos().contiguous(), emb.sin().contiguous()

def rotate_half(x: Tensor): return Tensor.cat(-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2], dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin): return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class LFM2ConvOperator:
    """ The actual convolution operator part of a conv block, now with caching. """
    def __init__(self, config: LFM2Config):
        self.hidden_size = config.hidden_size
        self.kernel_size = config.conv_kernel_size
        self.in_proj = Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.conv = Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1, # Causal padding
            groups=config.hidden_size,
            bias=False
        )
        self.out_proj = Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, x: Tensor, past_conv_state: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        B, C, x_proj = self.in_proj(x).chunk(3, dim=-1)
        x_gated = B * x_proj

        x_gated_permuted = x_gated.permute(0, 2, 1)

        if seq_len > 1:
            conv_out = self.conv(x_gated_permuted)[:, :, :seq_len]
            if seq_len < self.kernel_size:
                pad_amount = self.kernel_size - seq_len
                padding = Tensor.zeros(bsz, self.hidden_size, pad_amount, device=x.device, dtype=x.dtype)
                new_conv_state = Tensor.cat(padding, x_gated_permuted, dim=2)
            else:
                new_conv_state = x_gated_permuted[:, :, -self.kernel_size:]
        else: # Generation stage (seq_len == 1)
            assert past_conv_state is not None
            assert past_conv_state.shape[2] == self.kernel_size

            # Create the new state buffer by rolling: drop the oldest, append the newest.
            new_conv_state = Tensor.cat(past_conv_state[:, :, 1:], x_gated_permuted, dim=2)

            # Perform manual 1D convolution for the single step
            # Instead of calling self.conv() again, we do the dot product directly.
            # self.conv.weight shape: (hidden_size, 1, kernel_size) -> reshape for broadcasting
            conv_weights = self.conv.weight.reshape(1, self.hidden_size, self.kernel_size)
            # Element-wise product and sum over the kernel dimension.
            conv_out = (new_conv_state * conv_weights).sum(axis=2, keepdim=True)

        conv_out = conv_out.permute(0, 2, 1)
        output = self.out_proj(C * conv_out)
        return output, new_conv_state

class GroupedQueryAttention:
    """ Attention with QK Norm """
    def __init__(self, config: LFM2Config):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    # lfm2_modeling.py -> class GroupedQueryAttention

    def __call__(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        # --- MODIFIED ARGUMENT ---
        paged_kv_cache: PagedKVCache,
        cos_sin: Tuple[Tensor, Tensor],
        # --- NEW ARGUMENTS for paged attention ---
        start_pos: int,
        batch_idx: Tensor,
        seq_lens: List[int],
    ):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and apply norms
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)
        query_states = self.q_layernorm(query_states).permute(0, 2, 1, 3)
        key_states = self.k_layernorm(key_states).permute(0, 2, 1, 3)

        # Apply RoPE
        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # --- START: Paged Attention Logic ---
        
        # 1. Update the cache with the new key/value states
        # The shapes need to match what PagedKVCache.update expects
        # PagedKVCache expects (B, S, H, D)
        key_states_for_cache = key_states.permute(0, 2, 1, 3)
        value_states_for_cache = value_states.permute(0, 2, 1, 3)
        input_pos = Tensor.arange(start_pos, start_pos + q_len).reshape(1, q_len).expand(bsz, q_len)
        paged_kv_cache.update(batch_idx, input_pos, key_states_for_cache, value_states_for_cache)
        
        # 2. Gather all required keys/values for this attention operation
        # This will be slow because it gathers the full sequence every time.
        # More advanced implementations would only gather what's needed.
        # gather_kv_for_attention returns padded tensors.
        all_key_states, all_value_states = paged_kv_cache.gather_kv_for_attention(batch_idx, seq_lens)

        # Reshape gathered tensors for GQA and attention
        # all_key_states is (B, max_len, num_kv_heads, D)
        all_key_states = all_key_states.permute(0, 2, 1, 3) # -> (B, num_kv_heads, max_len, D)
        all_value_states = all_value_states.permute(0, 2, 1, 3) # -> (B, num_kv_heads, max_len, D)

        # GQA expansion
        all_key_states = all_key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        all_value_states = all_value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)

        # 3. Perform attention
        attn_output = Tensor.scaled_dot_product_attention(query_states, all_key_states, all_value_states, attn_mask=attention_mask)
        
        # --- END: Paged Attention Logic ---

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        
        # The new state is just the same cache object, updated in-place
        return self.out_proj(attn_output), paged_kv_cache

class SwiGLU:
    def __init__(self, hidden_size: int, intermediate_size: int):
        self.w1 = Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = Linear(intermediate_size, hidden_size, bias=False)
    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.w3(x))

class LFM2DecoderLayer:
    """ The new unified decoder layer for both Conv and Attention """
    def __init__(self, config: LFM2Config, is_attention_block: bool):
        self.feed_forward = SwiGLU(config.hidden_size, config.intermediate_size)
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_attention_block = is_attention_block

        if is_attention_block:
            self.operator = GroupedQueryAttention(config)
        else:
            self.operator = LFM2ConvOperator(config)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_state: Optional[Any], cos_sin: Tuple[Tensor, Tensor], start_pos: int = 0, batch_idx: Optional[Tensor] = None, seq_lens: Optional[List[int]] = None):
        residual = hidden_states

        if self.is_attention_block:
            # Pass the new arguments down to the attention operator
            hidden_states, new_state = self.operator(self.operator_norm(hidden_states), attention_mask, past_state, cos_sin, start_pos, batch_idx, seq_lens)
        else:
            hidden_states, new_state = self.operator(self.operator_norm(hidden_states), past_state)

        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states, new_state

class LFM2Model:
    def __init__(self, config: LFM2Config):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [LFM2DecoderLayer(config, i in config.full_attn_idxs) for i in range(config.num_hidden_layers)]
        
        # Pre-compute RoPE cache
        head_dim = config.hidden_size // config.num_attention_heads
        cos_cache, sin_cache = _precompute_rope_cache(
            dim=head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta
        )
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache


    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int = 0, output_hidden_states: bool = False, batch_idx: Optional[Tensor] = None, seq_lens: Optional[List[int]] = None):
        h = self.embed_tokens(input_ids)
        bsz, seq_len, _ = h.shape

        all_hidden_states = (h,) if output_hidden_states else None

        mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1).realize() if seq_len > 1 else None
        
        # Slice the pre-computed RoPE cache
        cos = self.cos_cache[start_pos : start_pos + seq_len].unsqueeze(0).expand(bsz, -1, -1)
        sin = self.sin_cache[start_pos : start_pos + seq_len].unsqueeze(0).expand(bsz, -1, -1)
        
        new_states_list = []
        current_h = h
        
        for i, layer in enumerate(self.layers):
            past_st = past_states[i] if past_states else None
            current_h, new_st = layer(current_h, mask, past_st, (cos, sin), start_pos, batch_idx, seq_lens) # Pass kwargs down
            new_states_list.append(new_st)
            
            # Apply the final norm INSIDE the final layer iteration
            # This replicates the exact structure from debug_prefilling.py that works.
            if i + 1 == len(self.layers):
                current_h = self.norm(current_h)
            
            if output_hidden_states:
                all_hidden_states += (current_h,)
        
        return current_h, new_states_list, all_hidden_states

class LFM2ForCausalLM:
    def __init__(self, config: LFM2Config):
        self.config = config
        self.model = LFM2Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.page_table = PageTable(
            n_pages=config.num_pages,
            page_size=config.page_size,
            max_batch_size=config.max_batch_size,
        )

        self.layer_caches = [None] * config.num_hidden_layers
        head_dim = config.hidden_size // config.num_attention_heads
        for i in range(config.num_hidden_layers):
            if i in config.full_attn_idxs:
                # All PagedKVCaches now share the same PageTable instance.
                self.layer_caches[i] = PagedKVCache(
                    page_table=self.page_table,
                    num_heads=config.num_key_value_heads,
                    head_dim=head_dim,
                    dtype=dtypes.float32,
                )

    def reset_request_state(self):
        """Resets the state for a new generation request."""
        # The PagedKVCache objects are persistent, but the conv caches are not.
        # We must clear the convolution tensors before starting a new prompt.
        for i in range(self.config.num_hidden_layers):
            if i not in self.config.full_attn_idxs:
                self.layer_caches[i] = None

    def __call__(self, input_ids: Tensor, start_pos: int = 0, output_hidden_states: bool = False, batch_idx: Optional[Tensor] = None, seq_lens: Optional[List[int]] = None) -> CausalLMOutputWithPast:
        assert batch_idx is not None and seq_lens is not None, "Paged attention requires batch_idx and seq_lens"

        # The model uses its own internal `self.layer_caches` as the past state.
        hidden_states, new_states, all_hidden_states = self.model(
            input_ids,
            past_states=self.layer_caches,
            start_pos=start_pos,
            output_hidden_states=output_hidden_states,
            batch_idx=batch_idx,
            seq_lens=seq_lens,
        )
        
        # *** NEW: Update the internal convolution caches ***
        # The PagedKVCaches are updated in-place, but the conv Tensors must be replaced.
        for i, state in enumerate(new_states):
            if i not in self.config.full_attn_idxs:
                self.layer_caches[i] = state

        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=self.layer_caches, # Return the updated caches
            hidden_states=all_hidden_states
        )

def _set_tensor(model: Any, key: str, tensor: Tensor):
    attrs = key.split('.')
    for i, attr in enumerate(attrs[:-1]):
        if isinstance(model, list): model = model[int(attr)]
        else: model = getattr(model, attr)
    final_attr = attrs[-1]
    if isinstance(model, list): model[int(final_attr)] = tensor
    else: setattr(model, final_attr, tensor)

def load_from_hf(model: LFM2ForCausalLM, repo_id: str, filename: str = "model.safetensors"):
    print(f"Fetching weights from {repo_id}/{filename}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    key_map = {
        "model.embedding_norm.weight": "model.norm.weight",
        "model.embed_tokens.weight": "model.embed_tokens.weight",
    }
    for i, layer in enumerate(model.model.layers):
        prefix_hf = f"model.layers.{i}"
        prefix_tg = f"model.layers.{i}"
        key_map.update({
            f"{prefix_hf}.operator_norm.weight": f"{prefix_tg}.operator_norm.weight",
            f"{prefix_hf}.ffn_norm.weight": f"{prefix_tg}.ffn_norm.weight",
            f"{prefix_hf}.feed_forward.w1.weight": f"{prefix_tg}.feed_forward.w1.weight",
            f"{prefix_hf}.feed_forward.w2.weight": f"{prefix_tg}.feed_forward.w2.weight",
            f"{prefix_hf}.feed_forward.w3.weight": f"{prefix_tg}.feed_forward.w3.weight",
        })
        if isinstance(layer.operator, GroupedQueryAttention):
            key_map.update({
                f"{prefix_hf}.self_attn.q_proj.weight": f"{prefix_tg}.operator.q_proj.weight",
                f"{prefix_hf}.self_attn.k_proj.weight": f"{prefix_tg}.operator.k_proj.weight",
                f"{prefix_hf}.self_attn.v_proj.weight": f"{prefix_tg}.operator.v_proj.weight",
                f"{prefix_hf}.self_attn.out_proj.weight": f"{prefix_tg}.operator.out_proj.weight",
                f"{prefix_hf}.self_attn.q_layernorm.weight": f"{prefix_tg}.operator.q_layernorm.weight",
                f"{prefix_hf}.self_attn.k_layernorm.weight": f"{prefix_tg}.operator.k_layernorm.weight",
            })
        elif isinstance(layer.operator, LFM2ConvOperator):
            key_map.update({
                f"{prefix_hf}.conv.in_proj.weight": f"{prefix_tg}.operator.in_proj.weight",
                f"{prefix_hf}.conv.conv.weight": f"{prefix_tg}.operator.conv.weight",
                f"{prefix_hf}.conv.out_proj.weight": f"{prefix_tg}.operator.out_proj.weight",
            })

    print("Loading and assigning weights...")
    GlobalCounters.reset()
    with safe_open(local_path, framework="pt", device="cpu") as f:
        for hf_key, tg_key in key_map.items():
            if hf_key not in f.keys():
                print(f"Warning: Weight key not found in safetensors file: {hf_key}")
                continue
            pt_tensor = f.get_tensor(hf_key)
            np_array = pt_tensor.to(torch.float32).numpy()
            tensor_tg = Tensor(np_array, requires_grad=False)
            _set_tensor(model, tg_key, tensor_tg)

    print("Re-tying word embeddings for lm_head...")
    model.lm_head.weight = model.model.embed_tokens.weight

    print("All weights loaded and assigned.")


def generate(
    model: LFM2ForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.8,
) -> str:
    Tensor.training = False
    print("\n--- Starting Text Generation ---")
    
    # Use the paged cache of the first attention layer as our controller
    first_attn_layer_idx = model.config.full_attn_idxs[0]
    paged_cache_controller = model.layer_caches[first_attn_layer_idx]
    if not isinstance(paged_cache_controller, PagedKVCache):
        raise RuntimeError("Model is not initialized with paged attention caches.")

    # *** NEW: Reset conv caches and then allocate pages ***
    model.reset_request_state()
    try:
        batch_idx_int = paged_cache_controller.page_table.allocate()
        print(f"Allocated batch slot: {batch_idx_int}")
        batch_idx_tensor = Tensor([batch_idx_int], dtype=dtypes.int32)
        
        messages = [{"role": "user", "content": prompt}]
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        tokens = list(prompt_tokens)
        
        max_len = len(tokens) + max_new_tokens
        paged_cache_controller.page_table.reserve(batch_idx_int, max_len)
        print(f"Reserved memory for a maximum sequence length of {max_len}")

        # --- Prefill Stage ---
        print("Processing prompt...")
        input_ids = Tensor([tokens])
        start_pos = 0
        
        # *** SIMPLIFIED: No more manual past_states management ***
        outputs = model(
            input_ids, 
            start_pos=start_pos,
            batch_idx=batch_idx_tensor,
            seq_lens=[len(tokens)]
        )
        logits = outputs.logits
        start_pos += len(tokens)

        # Sampling logic (unchanged)
        next_token_logits = logits[0, -1, :]
        if temperature == 0:
            next_token_id = next_token_logits.argmax().item()
        else:
            probs = (next_token_logits / temperature).softmax()
            sample = Tensor.uniform(1).item()
            next_token_id = (probs.cumsum() > sample).argmax().item()
        tokens.append(next_token_id)

        print("Generating new tokens...")
        print(tokenizer.decode(prompt_tokens), end="", flush=True)
        print(tokenizer.decode([next_token_id]), end="", flush=True)

        # --- Decoding Loop ---
        for _ in range(max_new_tokens - 1):
            input_ids = Tensor([[next_token_id]])
            
            # *** SIMPLIFIED: Model call is the same, state is managed internally ***
            outputs = model(
                input_ids,
                start_pos=start_pos,
                batch_idx=batch_idx_tensor,
                seq_lens=[start_pos + 1] 
            )
            logits = outputs.logits
            start_pos += 1

            # Sampling logic (unchanged)
            next_token_logits = logits[0, -1, :]
            if temperature == 0:
                next_token_id = next_token_logits.argmax().item()
            else:
                probs = (next_token_logits / temperature).softmax()
                sample = Tensor.uniform(1).item()
                next_token_id = (probs.cumsum() > sample).argmax().item()

            tokens.append(next_token_id)
            print(tokenizer.decode([next_token_id]), end="", flush=True)
            if next_token_id == tokenizer.eos_token_id:
                break

    finally:
        # Cleanup (unchanged)
        print(f"\nCleaning up and erasing batch slot: {batch_idx_int}")
        paged_cache_controller.page_table.erase(batch_idx_int)

    print("\n--- Generation Complete ---")
    return tokenizer.decode(tokens)