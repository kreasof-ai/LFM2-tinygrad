# FILE: src/model/gemma2_modeling.py

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type

# Third-party imports
from huggingface_hub import hf_hub_download

# tinygrad imports
from tinygrad import Tensor
from tinygrad.nn import Embedding, Linear

# Project imports
from model.base_modeling import BaseConfig, BaseForCausalLM
from utils.rope import _precompute_rope_cache, apply_rotary_pos_emb

# --- Configuration ---
@dataclass
class Gemma2Config(BaseConfig):
    """Configuration class for Gemma 2."""
    head_dim: int = 256
    hidden_act: str = "gelu_pytorch_tanh" # which is tinygrad's gelu
    attention_bias: bool = False
    
    # Gemma 2 specific
    query_pre_attn_scalar: float = 256.0
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0
    layer_types: List[str] = field(default_factory=list)
    sliding_window: Optional[int] = 4096

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "Gemma2Config":
        if config_dict.get("layer_types"):
            layer_types = config_dict["layer_types"]
        else:
            # Default logic for 2B/9B models: alternating sliding and full attention
            layer_types = [
                "sliding_attention" if (i + 1) % 2 != 0 else "full_attention"
                for i in range(config_dict["num_hidden_layers"])
            ]

        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict.get("num_key_value_heads"),
            head_dim=config_dict["head_dim"],
            rms_norm_eps=config_dict["rms_norm_eps"],
            rope_theta=config_dict["rope_theta"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
            # Gemma 2 specific
            query_pre_attn_scalar=config_dict["query_pre_attn_scalar"],
            attn_logit_softcapping=config_dict["attn_logit_softcapping"],
            final_logit_softcapping=config_dict["final_logit_softcapping"],
            layer_types=layer_types,
            sliding_window=config_dict.get("sliding_window"),
        )

# --- Model Components (Gemma2 Specific) ---

class Gemma2RMSNorm:
    """ Custom RMSNorm for Gemma2 with (1 + weight) scaling. """
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = Tensor.zeros(dim)

    def __call__(self, x: Tensor) -> Tensor:
        normed_x = x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return normed_x * (1.0 + self.weight)

class Gemma2MLP:
    """ Gemma2's MLP uses GELU activation. """
    def __init__(self, config: Gemma2Config, linear_class: Type = Linear):
        self.gate_proj = linear_class(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size, bias=False)
    def __call__(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x).gelu() * self.up_proj(x))

class Gemma2Attention:
    """
    Self-contained attention module for Gemma2. Does not use BaseAttention because it requires a custom
    attention scaling factor and logit softcapping.
    """
    def __init__(self, config: Gemma2Config, linear_class: Type):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.q_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = linear_class(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Any], cos_sin: Tuple[Tensor, Tensor], start_pos: int, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key_states = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)
        value_states = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)
        
        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_kv is not None:
            key_states = Tensor.cat(past_kv[0], key_states, dim=2)
            value_states = Tensor.cat(past_kv[1], value_states, dim=2)
        
        present_kv = (key_states, value_states)
        
        all_key_states = key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        all_value_states = value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        
        # Manual attention calculation
        attn_weights = (query_states @ all_key_states.transpose(-2, -1)) * self.scaling

        # Apply attn_logit_softcapping
        attn_weights = (attn_weights / self.config.attn_logit_softcapping).tanh() * self.config.attn_logit_softcapping

        if attention_mask is not None: attn_weights += attention_mask
        attn_weights = attn_weights.softmax(-1)
        attn_output = attn_weights @ all_value_states
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_kv

class Gemma2DecoderLayer:
    def __init__(self, config: Gemma2Config, linear_class: Type):
        self.self_attn = Gemma2Attention(config, linear_class)
        self.mlp = Gemma2MLP(config, linear_class)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Any], cos_sin: Tuple[Tensor, Tensor], start_pos: int, **kwargs):
        residual = hidden_states
        normed_hidden = self.input_layernorm(hidden_states)
        attn_output, new_kv = self.self_attn(normed_hidden, attention_mask, past_kv, cos_sin, start_pos, **kwargs)
        normed_attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + normed_attn_output
        
        residual = hidden_states
        normed_hidden = self.pre_feedforward_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden)
        normed_mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + normed_mlp_output
        return hidden_states, new_kv

class Gemma2Model:
    def __init__(self, config: Gemma2Config, linear_class: Type):
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma2DecoderLayer(config, linear_class) for _ in range(config.num_hidden_layers)]
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.cos_cache, self.sin_cache = _precompute_rope_cache(dim=config.head_dim, max_seq_len=config.max_position_embeddings, base=config.rope_theta, dtype=config.dtype)

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int, output_hidden_states: bool, **kwargs):
        h = self.embed_tokens(input_ids) * (self.config.hidden_size ** 0.5)
        bsz, seq_len, _ = h.shape
        all_hidden_states = (h,) if output_hidden_states else None

        causal_mask = Tensor.full((1, 1, seq_len, start_pos + seq_len), -float("inf")).triu(start_pos + 1).realize() if seq_len > 1 else None
        
        sliding_mask = None
        if self.config.sliding_window and seq_len > 1:
            q_pos = Tensor.arange(start_pos, start_pos + seq_len).reshape(seq_len, 1)
            k_pos = Tensor.arange(start_pos + seq_len).reshape(1, start_pos + seq_len)
            sliding_part = (q_pos - k_pos >= self.config.sliding_window).where(-float('inf'), 0)
            sliding_mask = (causal_mask + sliding_part).reshape(1, 1, seq_len, start_pos + seq_len).realize()
        
        cos = self.cos_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.config.head_dim).expand(bsz, 1, -1, -1)
        sin = self.sin_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.config.head_dim).expand(bsz, 1, -1, -1)
            
        new_states_list = []
        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            past_st = past_states[i] if past_states else None
            
            mask = sliding_mask if layer_type == "sliding_attention" and sliding_mask is not None else causal_mask
            
            h, new_st = layer(h, mask, past_st, (cos, sin), start_pos, **kwargs)
            new_states_list.append(new_st)
            if output_hidden_states: all_hidden_states += (h,)
        
        h = self.norm(h)
        if output_hidden_states: all_hidden_states += (h,)
        return h, new_states_list, all_hidden_states

class Gemma2ForCausalLM(BaseForCausalLM):
    def _create_model(self, config: BaseConfig, linear_class: Type) -> Any:
        return Gemma2Model(config, linear_class)

    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        # Apply final logit softcapping
        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            output.logits = (output.logits / softcap).tanh() * softcap
        return output

    @classmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        return Gemma2Config.from_hf_config(config_dict)
    
    def _get_key_map(self) -> dict:
        key_map = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i in range(self.config.num_hidden_layers):
            p = f"model.layers.{i}"
            key_map.update({
                f"{p}.input_layernorm.weight": f"{p}.input_layernorm.weight",
                f"{p}.self_attn.q_proj.weight": f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight": f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight": f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight": f"{p}.self_attn.o_proj.weight",
                f"{p}.post_attention_layernorm.weight": f"{p}.post_attention_layernorm.weight",
                f"{p}.pre_feedforward_layernorm.weight": f"{p}.pre_feedforward_layernorm.weight",
                f"{p}.mlp.gate_proj.weight": f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight": f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight": f"{p}.mlp.down_proj.weight",
                f"{p}.post_feedforward_layernorm.weight": f"{p}.post_feedforward_layernorm.weight",
            })
        return key_map