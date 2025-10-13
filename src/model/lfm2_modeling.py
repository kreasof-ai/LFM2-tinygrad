# src/model/lfm2_modeling.py

"""
LFM2 (Liquid Foundation Model 2) tinygrad Implementation

This is a unified implementation supporting:
1. Standard float32 inference.
2. Experimental Paged Attention for efficient KV cache management.
3. float16 precision for training and inference.

The behavior is controlled by flags in the LFM2Config class.
"""


import json
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type, Union

# Third-party imports
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch # only used for converting weights
from transformers import AutoTokenizer

# tinygrad imports
from tinygrad import Tensor, dtypes, GlobalCounters, Device
from tinygrad.helpers import getenv
from tinygrad.nn import Conv1d, Embedding, Linear, RMSNorm
from tinygrad.nn.state import load_state_dict

# Project imports for Paged Attention
from experimental.paged_attention import PagedKVCache, PageTable
from extra.quantization import NF4Linear

# --- HF-like Model Output ---

@dataclass
class CausalLMOutputWithPast:
    """
    Base class for causal language model (or autoregressive) outputs.
    Adapted for tinygrad from Hugging Face's CausalLMOutputWithPast.
    """
    logits: Tensor
    loss: Optional[Tensor] = None
    past_key_values: Optional[List[Any]] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None

# --- Configuration ---

@dataclass
class LFM2Config:
    """Unified configuration class for LFM2."""
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

    # --- FUSION FLAGS ---
    use_paged_attention: bool = False
    dtype: Any = dtypes.float32

    # --- Quantization ---
    quantize: Optional[str] = None

    # --- Paged Attention specific fields ---
    page_size: int = 16
    max_batch_size: int = 8
    num_pages: int = 2048

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

def _precompute_rope_cache(dim: int, max_seq_len: int, base: float, dtype) -> Tuple[Tensor, Tensor]:
    """Pre-computes the rotary positional embeddings for the given dimensions."""
    inv_freq = 1.0 / (base ** (Tensor.arange(0, dim, 2, dtype=dtype) / dim))
    t = Tensor.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = t.reshape(-1, 1) * inv_freq.reshape(1, -1)
    emb = Tensor.cat(freqs, freqs, dim=-1)
    return emb.cos().contiguous(), emb.sin().contiguous()

def rotate_half(x: Tensor): return Tensor.cat(-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2], dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin): return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class LFM2ConvOperator:
    """ The convolution operator part of a conv block, with caching. """
    def __init__(self, config: LFM2Config, linear_class: Type = Linear):
        self.hidden_size = config.hidden_size
        self.kernel_size = config.conv_kernel_size
        self.in_proj = linear_class(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.conv = Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1, # Causal padding
            groups=config.hidden_size,
            bias=False
        )
        self.out_proj = linear_class(config.hidden_size, config.hidden_size, bias=False)
        
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
            new_conv_state = Tensor.cat(past_conv_state[:, :, 1:], x_gated_permuted, dim=2)
            conv_weights = self.conv.weight.reshape(1, self.hidden_size, self.kernel_size)
            conv_out = (new_conv_state * conv_weights).sum(axis=2, keepdim=True)

        conv_out = conv_out.permute(0, 2, 1)
        output = self.out_proj(C * conv_out)
        return output, new_conv_state

class GroupedQueryAttention:
    """ Attention with QK Norm, supporting both standard and paged KV caching. """
    def __init__(self, config: LFM2Config, linear_class: Type = Linear):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = linear_class(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Any], cos_sin: Tuple[Tensor, Tensor], start_pos: int, batch_idx: Optional[Tensor], seq_lens: Optional[List[int]]):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_layernorm(query_states).permute(0, 2, 1, 3)
        key_states = self.k_layernorm(key_states).permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)

        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config.use_paged_attention:
            paged_kv_cache = past_kv
            input_pos = Tensor.arange(start_pos, start_pos + q_len).reshape(1, q_len).expand(bsz, q_len)
            paged_kv_cache.update(batch_idx, input_pos, key_states.permute(0, 2, 1, 3), value_states.permute(0, 2, 1, 3))
            
            all_key_states, all_value_states = paged_kv_cache.gather_kv_for_attention(batch_idx, seq_lens)
            all_key_states = all_key_states.permute(0, 2, 1, 3)
            all_value_states = all_value_states.permute(0, 2, 1, 3)
            present_kv = paged_kv_cache
        else:
            past_key_value = past_kv
            if past_key_value is not None:
                key_states = Tensor.cat(past_key_value[0], key_states, dim=2)
                value_states = Tensor.cat(past_key_value[1], value_states, dim=2)
            all_key_states, all_value_states = key_states, value_states
            present_kv = (key_states, value_states)

        all_key_states = all_key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        all_value_states = all_value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        
        attn_output = Tensor.scaled_dot_product_attention(query_states, all_key_states, all_value_states, attn_mask=attention_mask)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        return self.out_proj(attn_output), present_kv

class SwiGLU:
    def __init__(self, hidden_size: int, intermediate_size: int, linear_class: Type = Linear):
        self.w1 = linear_class(hidden_size, intermediate_size, bias=False)
        self.w3 = linear_class(hidden_size, intermediate_size, bias=False)
        self.w2 = linear_class(intermediate_size, hidden_size, bias=False)
    def __call__(self, x: Tensor) -> Tensor:
        w1 = self.w1(x).silu()
        w3 = self.w3(x.contiguous_backward())  # this fixes a strange fusion that makes tensor cores miss
        return self.w2(w1 * w3)

class LFM2DecoderLayer:
    def __init__(self, config: LFM2Config, is_attention_block: bool, linear_class: Type = Linear):
        self.config = config
        self.feed_forward = SwiGLU(config.hidden_size, config.intermediate_size, linear_class)
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_attention_block = is_attention_block
        self.operator = GroupedQueryAttention(config, linear_class) if is_attention_block else LFM2ConvOperator(config, linear_class)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_state: Optional[Any], cos_sin: Tuple[Tensor, Tensor], start_pos: int, batch_idx: Optional[Tensor], seq_lens: Optional[List[int]]):
        residual = hidden_states
        normed_hidden = self.operator_norm(hidden_states)
        
        if self.is_attention_block:
            hidden_states, new_state = self.operator(normed_hidden, attention_mask, past_state, cos_sin, start_pos, batch_idx, seq_lens)
        else:
            hidden_states, new_state = self.operator(normed_hidden, past_state)

        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states, new_state

class LFM2Model:
    def __init__(self, config: LFM2Config, linear_class: Type = Linear):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [LFM2DecoderLayer(config, i in config.full_attn_idxs, linear_class) for i in range(config.num_hidden_layers)]
        
        head_dim = config.hidden_size // config.num_attention_heads
        cos_cache, sin_cache = _precompute_rope_cache(dim=head_dim, max_seq_len=config.max_position_embeddings, base=config.rope_theta, dtype=config.dtype)
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int, output_hidden_states: bool, batch_idx: Optional[Tensor], seq_lens: Optional[List[int]]):
        h = self.embed_tokens(input_ids)
        bsz, seq_len, _ = h.shape
        all_hidden_states = (h,) if output_hidden_states else None
        mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1).realize() if seq_len > 1 else None
        
        head_dim = self.cos_cache.shape[-1]
        cos = self.cos_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, head_dim).expand(bsz, 1, seq_len, head_dim)
        sin = self.sin_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, head_dim).expand(bsz, 1, seq_len, head_dim)
        
        new_states_list = []
        current_h = h
        
        for i, layer in enumerate(self.layers):
            past_st = past_states[i] if past_states else None
            current_h, new_st = layer(current_h, mask, past_st, (cos, sin), start_pos, batch_idx, seq_lens)
            new_states_list.append(new_st)
            
            if i + 1 == len(self.layers):
                current_h = self.norm(current_h)
            
            if output_hidden_states:
                all_hidden_states += (current_h,)
        
        return current_h, new_states_list, all_hidden_states

class LFM2ForCausalLM:
    def __init__(self, config: LFM2Config):
        self.config = config
        
        if config.quantize == "nf4":
            self.linear_class = NF4Linear() # Use default block_size=64
        else:
            self.linear_class = Linear

        self.model = LFM2Model(config, self.linear_class)
        # lm_head is typically not quantized
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        if config.use_paged_attention:
            self.page_table = PageTable(n_pages=config.num_pages, page_size=config.page_size, max_batch_size=config.max_batch_size)
            self.layer_caches = [None] * config.num_hidden_layers
            head_dim = config.hidden_size // config.num_attention_heads
            for i in range(config.num_hidden_layers):
                if i in config.full_attn_idxs:
                    self.layer_caches[i] = PagedKVCache(page_table=self.page_table, num_heads=config.num_key_value_heads, head_dim=head_dim, dtype=config.dtype)

    def reset_request_state(self):
        """Resets convolution caches for a new paged generation request."""
        assert self.config.use_paged_attention, "This method is for paged attention only."
        for i in range(self.config.num_hidden_layers):
            if i not in self.config.full_attn_idxs:
                self.layer_caches[i] = None

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]] = None, start_pos: int = 0, labels: Optional[Tensor] = None, output_hidden_states: bool = False, batch_idx: Optional[Tensor] = None, seq_lens: Optional[List[int]] = None) -> CausalLMOutputWithPast:
        if self.config.use_paged_attention:
            assert batch_idx is not None and seq_lens is not None, "Paged attention requires batch_idx and seq_lens"
            _past_states = self.layer_caches
        else:
            _past_states = past_states

        hidden_states, new_states, all_hidden_states = self.model(input_ids, _past_states, start_pos, output_hidden_states, batch_idx, seq_lens)
        
        if self.config.use_paged_attention:
            for i, state in enumerate(new_states):
                if i not in self.config.full_attn_idxs: self.layer_caches[i] = state
        
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = logits[..., :-1, :].flatten(0, 1).sparse_categorical_crossentropy(labels[..., 1:].flatten(), ignore_index=-100)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_states if not self.config.use_paged_attention else self.layer_caches,
            hidden_states=all_hidden_states
        )

def _set_tensor(model: Any, key: str, tensor: Tensor):
    attrs = key.split('.')
    target = model
    for attr in attrs[:-1]:
        target = getattr(target, attr) if not attr.isdigit() else target[int(attr)]
    setattr(target, attrs[-1], tensor)

def load_from_hf(model: LFM2ForCausalLM, repo_id: str, filename: str = "model.safetensors"):
    """
    Loads weights from a Hugging Face safetensors file into the tinygrad model.
    Handles on-the-fly quantization if configured.
    """
    print(f"Fetching weights from {repo_id}/{filename}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    dtype = model.config.dtype

    # Define mapping from Hugging Face weight names to our tinygrad model's attribute names
    key_map = {
        "model.embedding_norm.weight": "model.norm.weight",
        "model.embed_tokens.weight": "model.embed_tokens.weight"
    }
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}"
        key_map.update({
            f"{p}.operator_norm.weight": f"{p}.operator_norm.weight",
            f"{p}.ffn_norm.weight": f"{p}.ffn_norm.weight",
            f"{p}.feed_forward.w1.weight": f"{p}.feed_forward.w1.weight",
            f"{p}.feed_forward.w2.weight": f"{p}.feed_forward.w2.weight",
            f"{p}.feed_forward.w3.weight": f"{p}.feed_forward.w3.weight"
        })
        if isinstance(layer.operator, GroupedQueryAttention):
            key_map.update({
                f"{p}.self_attn.q_proj.weight": f"{p}.operator.q_proj.weight",
                f"{p}.self_attn.k_proj.weight": f"{p}.operator.k_proj.weight",
                f"{p}.self_attn.v_proj.weight": f"{p}.operator.v_proj.weight",
                f"{p}.self_attn.out_proj.weight": f"{p}.operator.out_proj.weight",
                f"{p}.self_attn.q_layernorm.weight": f"{p}.operator.q_layernorm.weight", 
                f"{p}.self_attn.k_layernorm.weight": f"{p}.operator.k_layernorm.weight"
            })
        elif isinstance(layer.operator, LFM2ConvOperator):
            key_map.update({
                f"{p}.conv.in_proj.weight": f"{p}.operator.in_proj.weight",
                f"{p}.conv.conv.weight": f"{p}.operator.conv.weight",
                f"{p}.conv.out_proj.weight": f"{p}.operator.out_proj.weight"
            })

    # 1. Load all weights into a temporary dictionary with tinygrad names
    tg_state_dict = {}
    print("Loading weights into memory...")
    with safe_open(local_path, framework="pt", device="cpu") as f:
        for hf_key, tg_key in key_map.items():
            if hf_key not in f.keys():
                print(f"Warning: Weight key not found in safetensors file: {hf_key}")
                continue
            pt_tensor = f.get_tensor(hf_key)
            # Load as float32 first for quantization stability
            np_array = pt_tensor.to(torch.float32).numpy()
            tg_state_dict[tg_key] = Tensor(np_array, requires_grad=False)

    # 2. If quantization is enabled, transform the state dictionary
    if model.config.quantize == "nf4":
        assert hasattr(model, 'linear_class') and hasattr(model.linear_class, 'quantize'), \
            "Model must be initialized with a quantizable linear class."
        device = getattr(model.model.embed_tokens.weight, 'device', Device.DEFAULT)
        tg_state_dict = model.linear_class.quantize(tg_state_dict, device=device)

    # 3. Cast non-quantized weights to the model's target dtype
    for k in tg_state_dict:
        if tg_state_dict[k].dtype != dtypes.uint8: # Don't change quantized weights
            tg_state_dict[k] = tg_state_dict[k].cast(dtype)

    # 4. Load the final state dictionary into the model
    print("Assigning weights to model...")
    load_state_dict(model, tg_state_dict, strict=False)

    if model.config.tie_word_embeddings:
        print("Re-tying word embeddings for lm_head...")
        model.lm_head.weight = model.model.embed_tokens.weight
    print("All weights loaded and assigned.")


def generate(model: LFM2ForCausalLM, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int, temperature: float = 0.8) -> str:
    Tensor.training = False
    print("\n--- Starting Text Generation ---")
    
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    tokens = list(prompt_tokens)

    def sample(logits: Tensor) -> int:
        if temperature == 0: return logits.argmax().item()
        probs = (logits / temperature).softmax()
        return (probs.cumsum() > Tensor.uniform(1).item()).argmax().item()

    def decode_one_token(next_token_id: int):
        tokens.append(next_token_id)
        print(tokenizer.decode([next_token_id]), end="", flush=True)

    if model.config.use_paged_attention:
        batch_idx_int = -1
        try:
            model.reset_request_state()
            batch_idx_int = model.page_table.allocate()
            batch_idx_tensor = Tensor([batch_idx_int], dtype=dtypes.int32)
            max_len = len(tokens) + max_new_tokens
            model.page_table.reserve(batch_idx_int, max_len)

            print("Processing prompt...")
            outputs = model(Tensor([tokens]), start_pos=0, batch_idx=batch_idx_tensor, seq_lens=[len(tokens)])
            start_pos = len(tokens)
            next_token = sample(outputs.logits[0, -1, :])
            
            print("Generating new tokens...")
            print(tokenizer.decode(prompt_tokens), end="", flush=True)
            decode_one_token(next_token)

            for _ in range(max_new_tokens - 1):
                outputs = model(Tensor([[next_token]]), start_pos=start_pos, batch_idx=batch_idx_tensor, seq_lens=[start_pos + 1])
                start_pos += 1
                next_token = sample(outputs.logits[0, -1, :])
                decode_one_token(next_token)
                if next_token == tokenizer.eos_token_id: break
        finally:
            if batch_idx_int != -1: model.page_table.erase(batch_idx_int)
    else: # Standard generation
        past_states = [None] * len(model.model.layers)
        print("Processing prompt...")
        outputs = model(Tensor([tokens]), past_states, start_pos=0)
        start_pos = len(tokens)
        past_states = outputs.past_key_values
        next_token = sample(outputs.logits[0, -1, :])

        print("Generating new tokens...")
        print(tokenizer.decode(prompt_tokens), end="", flush=True)
        decode_one_token(next_token)

        for _ in range(max_new_tokens - 1):
            outputs = model(Tensor([[next_token]]), past_states, start_pos=start_pos)
            start_pos += 1
            past_states = outputs.past_key_values
            next_token = sample(outputs.logits[0, -1, :])
            decode_one_token(next_token)
            if next_token == tokenizer.eos_token_id: break
    
    print("\n--- Generation Complete ---")
    return tokenizer.decode(tokens)