# Modify this file: src/model/hunyuan_modeling.py

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type

# Third-party imports
from huggingface_hub import hf_hub_download

# tinygrad imports
from tinygrad import Tensor
from tinygrad.nn import RMSNorm, Embedding

# Project imports for shared components
from model.base_modeling import BaseConfig, BaseAttention, BaseMLP, BaseModel, BaseForCausalLM
from utils.rope import _precompute_rope_cache, _precompute_rope_cache_dynamic

# --- Configuration ---
@dataclass
class HunyuanConfig(BaseConfig):
    """Configuration class for Hunyuan."""
    head_dim: int = 128
    hidden_act: str = "silu"
    qk_norm: bool = True # Hunyuan uses QK Norm
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_scaling: Optional[dict] = field(default=None)

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "HunyuanConfig":
        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            head_dim=config_dict["head_dim"],
            rms_norm_eps=config_dict["rms_norm_eps"],
            rope_theta=config_dict["rope_theta"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
            qk_norm=config_dict.get("use_qk_norm", False),
            attention_bias=config_dict.get("attention_bias", False),
            mlp_bias=config_dict.get("mlp_bias", False),
            rope_scaling=config_dict.get("rope_scaling"),
        )

# --- Model Components ---
class HunyuanDecoderLayer:
    def __init__(self, config: HunyuanConfig, linear_class: Type):
        self.self_attn = BaseAttention(config, linear_class)
        self.mlp = BaseMLP(config, linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Tuple[Tensor, Tensor]], cos_sin: Tuple[Tensor, Tensor], start_pos: int, **kwargs):
        residual = hidden_states
        normed_hidden = self.input_layernorm(hidden_states)
        attn_output, new_kv = self.self_attn(normed_hidden, attention_mask, past_kv, cos_sin, start_pos, **kwargs)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        normed_hidden = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden)
        hidden_states = residual + mlp_output
        return hidden_states, new_kv

class HunyuanModel(BaseModel):
    def __init__(self, config: HunyuanConfig, linear_class: Type):
        # Override BaseModel's __init__ to select the correct RoPE function
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [self._create_decoder_layer(config, linear_class) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.head_dim = config.head_dim

        # Select the correct RoPE precomputation function
        if config.rope_scaling and config.rope_scaling.get("type") == "dynamic":
            print("Using dynamic RoPE.")
            # Add theta to the dict for the func to use
            config.rope_scaling['rope_theta'] = config.rope_theta 
            cos_cache, sin_cache = _precompute_rope_cache_dynamic(
                dim=self.head_dim, 
                max_seq_len=config.max_position_embeddings, 
                rope_scaling=config.rope_scaling,
                dtype=config.dtype
            )
        else:
            print("Warning: Hunyuan model detected but not using dynamic RoPE. Falling back to standard RoPE.")
            cos_cache, sin_cache = _precompute_rope_cache(
                dim=self.head_dim, 
                max_seq_len=config.max_position_embeddings, 
                base=config.rope_theta, 
                dtype=config.dtype
            )
        
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    def _create_decoder_layer(self, config: BaseConfig, linear_class: Type):
        return HunyuanDecoderLayer(config, linear_class)

class HunyuanForCausalLM(BaseForCausalLM):
    def _create_model(self, config: BaseConfig, linear_class: Type) -> BaseModel:
        return HunyuanModel(config, linear_class)
    
    @classmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        return HunyuanConfig.from_hf_config(config_dict)
    
    def _get_key_map(self) -> dict:
        key_map = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i in range(self.config.num_hidden_layers):
            p = f"model.layers.{i}"
            # HF uses `query_layernorm` and `key_layernorm`, my BaseAttention uses `q_norm` and `k_norm`
            key_map.update({
                f"{p}.input_layernorm.weight": f"{p}.input_layernorm.weight",
                f"{p}.post_attention_layernorm.weight": f"{p}.post_attention_layernorm.weight",
                f"{p}.mlp.gate_proj.weight": f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight": f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight": f"{p}.mlp.down_proj.weight",
                f"{p}.self_attn.q_proj.weight": f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight": f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight": f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight": f"{p}.self_attn.o_proj.weight",
                f"{p}.self_attn.query_layernorm.weight": f"{p}.self_attn.q_norm.weight",
                f"{p}.self_attn.key_layernorm.weight": f"{p}.self_attn.k_norm.weight",
            })
        return key_map