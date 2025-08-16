"""
LFM2 (Liquid Foundation Model 2) tinygrad Implementation

This version is adapted to load pretrained weights from Hugging Face Hub.

Heavily inspired from https://github.com/kyegomez/LFM2 and official https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py implementation
"""


import json
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# Third-party imports
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch # only used for converting weights
from tqdm import trange
from transformers import AutoTokenizer

# tinygrad imports
from tinygrad import Tensor, dtypes, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.nn import Conv2d, Embedding, Linear, RMSNorm

# For reproducible tests
if getenv("SEED"):
    Tensor.manual_seed(getenv("SEED"))

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

class RotaryPositionalEmbedding:
    def __init__(self, dim: int, base: float):
        inv_freq = 1.0 / (base ** (Tensor.arange(0, dim, 2, dtype=dtypes.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, x: Tensor, seq_len: int, start_pos: int = 0) -> Tuple[Tensor, Tensor]:
        t = Tensor.arange(start_pos, start_pos + seq_len, device=x.device).cast(self.inv_freq.dtype)
        freqs = t.reshape(-1, 1) * self.inv_freq.reshape(1, -1)
        emb = Tensor.cat(freqs, freqs, dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x: Tensor): return Tensor.cat(-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2], dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin): return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class LFM2ConvOperator:
    """ The actual convolution operator part of a conv block """
    def __init__(self, config: LFM2Config):
        self.in_proj = Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.conv = Conv2d(config.hidden_size, config.hidden_size,
                           kernel_size=(config.conv_kernel_size, 1),
                           padding=0,
                           groups=config.hidden_size, bias=False)
        self.out_proj = Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        B, C, x_proj = self.in_proj(x).chunk(3, dim=-1)
        x_gated = B * x_proj.silu()

        x_conv = x_gated.permute(0, 2, 1).unsqueeze(3)
        pad_arg = ((0, 0), (0, 0), (self.conv.kernel_size[0] - 1, 0), (0, 0))
        x_conv = x_conv.pad(pad_arg)
        x_conv = self.conv(x_conv).squeeze(3).permute(0, 2, 1)

        return self.out_proj(C * x_conv)

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

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_key_value: Optional[Tuple[Tensor]], cos_sin: Tuple[Tensor, Tensor]):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)

        query_states = self.q_layernorm(query_states).permute(0, 2, 1, 3)
        key_states = self.k_layernorm(key_states).permute(0, 2, 1, 3)

        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = Tensor.cat(past_key_value[0], key_states, dim=2)
            value_states = Tensor.cat(past_key_value[1], value_states, dim=2)
        past_key_value = (key_states, value_states)

        key_states = key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        value_states = value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)

        attn_weights = (query_states @ key_states.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if attention_mask is not None: attn_weights += attention_mask

        attn_weights = attn_weights.softmax(-1).cast(query_states.dtype)
        attn_output = (attn_weights @ value_states).permute(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        return self.out_proj(attn_output), past_key_value

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

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_key_value: Optional[Tuple[Tensor]], cos_sin: Tuple[Tensor, Tensor]):
        residual = hidden_states
        normed_h = self.operator_norm(hidden_states)

        if self.is_attention_block:
            operator_out, new_kv_cache = self.operator(normed_h, attention_mask, past_key_value, cos_sin)
        else:
            operator_out = self.operator(normed_h)
            new_kv_cache = None

        h = residual + operator_out
        residual = h
        normed_h = self.ffn_norm(h)
        ffn_out = self.feed_forward(normed_h)
        h = residual + ffn_out

        return h, new_kv_cache

class LFM2Model:
    def __init__(self, config: LFM2Config):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = []
        for i in range(config.num_hidden_layers):
            is_attention = i in config.full_attn_idxs
            self.layers.append(LFM2DecoderLayer(config, is_attention_block=is_attention))
            print(f"Layer {i}: {'LFM2DecoderLayer (Attention)' if is_attention else 'LFM2DecoderLayer (Convolution)'}")
        self.rotary_emb = RotaryPositionalEmbedding(config.hidden_size // config.num_attention_heads, config.rope_theta)

    def __call__(self, input_ids: Tensor, past_key_values: Optional[List[Tuple[Tensor]]], start_pos: int = 0):
        h = self.embed_tokens(input_ids)
        seq_len = h.shape[1]

        mask = None
        if seq_len > 1:
            mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1).realize()

        cos_sin = self.rotary_emb(h, seq_len, start_pos)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            h, new_kv = layer(h, mask, past_kv, cos_sin)
            new_kv_caches.append(new_kv)

        h = self.norm(h)

        return h, new_kv_caches

class LFM2ForCausalLM:
    def __init__(self, config: LFM2Config):
        self.model = LFM2Model(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(self, input_ids: Tensor, past_key_values: Optional[List[Tuple[Tensor]]] = None, start_pos: int = 0):
        hidden_states, new_kv_caches = self.model(input_ids, past_key_values, start_pos)
        return self.lm_head(hidden_states), new_kv_caches

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
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.embedding_norm.weight": "model.norm.weight",
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
            if 'conv.conv.weight' in hf_key:
                tensor_tg = tensor_tg.unsqueeze(-1)
            _set_tensor(model, tg_key, tensor_tg)
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
    print(f"Prompt: {prompt}")

    prompt_tokens = tokenizer.encode(prompt)
    if tokenizer.bos_token_id and prompt_tokens[0] != tokenizer.bos_token_id:
        prompt_tokens = [tokenizer.bos_token_id] + prompt_tokens

    tokens = [t for t in prompt_tokens]
    input_ids = Tensor([tokens])
    past_key_values = [None] * len(model.model.layers)
    start_pos = 0

    print("Processing prompt...")
    logits, past_key_values = model(input_ids, past_key_values, start_pos=start_pos)
    start_pos += len(tokens)

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

    for _ in trange(max_new_tokens - 1):
        input_ids = Tensor([[next_token_id]])
        logits, past_key_values = model(input_ids, past_key_values, start_pos=start_pos)
        start_pos += 1

        next_token_logits = logits[0, -1, :]
        if temperature == 0:
            next_token_id = next_token_logits.argmax().item()
        else:
            probs = (next_token_logits / temperature).softmax()
            sample = Tensor.uniform(1).item()
            next_token_id = (probs.cumsum() > sample).argmax().item()

        tokens.append(next_token_id)
        # print(tokenizer.decode([next_token_id]), end="", flush=True)
        if next_token_id == tokenizer.eos_token_id:
            break

    print("\n--- Generation Complete ---")
    return tokenizer.decode(tokens)


if __name__ == "__main__":
    REPO_ID = "LiquidAI/LFM2-350M"
    print(f"--- Loading LFM2 Model: {REPO_ID} ---")

    # 1. Download and load the configuration
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = LFM2Config.from_hf_config(config_dict)
    print("\nModel configuration:")
    print(config)

    # 2. Initialize the model structure
    print("\nInitializing model architecture...")
    model = LFM2ForCausalLM(config)

    # 3. Load the pretrained weights
    load_from_hf(model, REPO_ID, filename="model.safetensors")

    # 4. Load the tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    # 5. Run text generation
    prompt = "The secret to a long and happy life is"
    generated_text = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=20,
        temperature=0.3
    )

    print("\nFinal generated text:")
    print(generated_text)