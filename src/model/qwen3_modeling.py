# src/model/qwen3_modeling.py

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type

# Third-party imports
from huggingface_hub import hf_hub_download

# tinygrad imports
from tinygrad import Tensor
from tinygrad.nn import RMSNorm

# Project imports for shared components
from model.base_modeling import BaseConfig, BaseAttention, BaseMLP, BaseModel, BaseForCausalLM

# --- Configuration ---
@dataclass
class Qwen3Config(BaseConfig):
    """Configuration class for Qwen3."""
    head_dim: int = 128
    hidden_act: str = "silu"
    qk_norm: bool = True # Custom flag to indicate QK norm is used

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "Qwen3Config":
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
        )

# --- Model Components ---
class Qwen3DecoderLayer:
    def __init__(self, config: Qwen3Config, linear_class: Type):
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

class Qwen3Model(BaseModel):
    def _create_decoder_layer(self, config: BaseConfig, linear_class: Type):
        return Qwen3DecoderLayer(config, linear_class)

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int, output_hidden_states: bool, **kwargs):
        # Qwen3's norm is applied before the final lm_head, so we call super and then apply norm.
        h, new_states_list, all_hidden_states = super().__call__(input_ids, past_states, start_pos, output_hidden_states, **kwargs)
        # Note: The base class already applies the final norm. This is correct for Qwen3.
        return h, new_states_list, all_hidden_states

class Qwen3ForCausalLM(BaseForCausalLM):
    def _create_model(self, config: BaseConfig, linear_class: Type) -> BaseModel:
        return Qwen3Model(config, linear_class)
    
    @classmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        return Qwen3Config.from_hf_config(config_dict)
    
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
                f"{p}.post_attention_layernorm.weight": f"{p}.post_attention_layernorm.weight",
                f"{p}.mlp.gate_proj.weight": f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight": f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight": f"{p}.mlp.down_proj.weight",
                f"{p}.self_attn.q_proj.weight": f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight": f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight": f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight": f"{p}.self_attn.o_proj.weight",
                f"{p}.self_attn.q_norm.weight": f"{p}.self_attn.q_norm.weight",
                f"{p}.self_attn.k_norm.weight": f"{p}.self_attn.k_norm.weight",
            })
        return key_map