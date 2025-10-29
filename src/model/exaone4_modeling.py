# src/model/exaone4_modeling.py

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
from utils.rope import _precompute_rope_cache_llama3

# --- Configuration ---
@dataclass
class Exaone4Config(BaseConfig):
    """Configuration class for Exaone4."""
    hidden_act: str = "silu"
    qk_norm: bool = True
    attention_bias: bool = False
    rope_scaling: Optional[dict] = field(default=None)
    sliding_window: Optional[int] = None
    layer_types: List[str] = field(default_factory=list)

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "Exaone4Config":
        head_dim = config_dict.get("head_dim", config_dict["hidden_size"] // config_dict["num_attention_heads"])
        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            head_dim=head_dim,
            rms_norm_eps=config_dict["rms_norm_eps"],
            rope_theta=config_dict["rope_theta"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
            rope_scaling=config_dict.get("rope_scaling"),
            sliding_window=config_dict.get("sliding_window"),
            layer_types=config_dict.get("layer_types", [])
        )

# --- Model Components ---
class Exaone4DecoderLayer:
    """ Implements the custom post-normalization decoder layer for Exaone4. """
    def __init__(self, config: Exaone4Config, linear_class: Type):
        self.self_attn = BaseAttention(config, linear_class)
        self.mlp = BaseMLP(config, linear_class)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Tuple[Tensor, Tensor]], cos_sin: Tuple[Tensor, Tensor], start_pos: int, **kwargs):
        # Self Attention block with post-normalization
        residual = hidden_states
        attn_output, new_kv = self.self_attn(hidden_states, attention_mask, past_kv, cos_sin, start_pos, **kwargs)
        normed_attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + normed_attn_output

        # MLP block with post-normalization
        residual = hidden_states
        mlp_output = self.mlp(hidden_states)
        normed_mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + normed_mlp_output
        return hidden_states, new_kv

class Exaone4Model(BaseModel):
    def __init__(self, config: Exaone4Config, linear_class: Type):
        # We don't call super().__init__ to handle the specific RoPE initialization
        self.config = config # Store config for __call__
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [self._create_decoder_layer(config, linear_class) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head_dim = config.head_dim

        # Use Llama3-style RoPE scaling as specified in the config
        assert config.rope_scaling and config.rope_scaling.get("rope_type") == "llama3", "Exaone4 requires Llama3-style RoPE scaling."
        print("Using Llama 3 specific RoPE scaling for Exaone4.")
        config.rope_scaling['rope_theta'] = config.rope_theta # Add theta for the func
        cos_cache, sin_cache = _precompute_rope_cache_llama3(
            dim=self.head_dim,
            max_seq_len=config.max_position_embeddings,
            rope_scaling=config.rope_scaling,
            dtype=config.dtype
        )
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    def _create_decoder_layer(self, config: BaseConfig, linear_class: Type):
        return Exaone4DecoderLayer(config, linear_class)

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int, output_hidden_states: bool, **kwargs):
        """ Overridden __call__ to handle hybrid attention masking. """
        h = self.embed_tokens(input_ids)
        bsz, seq_len, _ = h.shape
        all_hidden_states = (h,) if output_hidden_states else None

        # Create masks for full and sliding window attention
        full_mask, sliding_mask = None, None
        if seq_len > 1:
            full_mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1)
            # Only create a sliding mask if the model uses it
            if self.config.sliding_window and "sliding_attention" in self.config.layer_types:
                q_pos = Tensor.arange(seq_len).reshape(seq_len, 1)
                k_pos = Tensor.arange(seq_len).reshape(1, seq_len)
                # Mask out positions outside the sliding window
                sliding_part = (q_pos - k_pos >= self.config.sliding_window).where(-float('inf'), 0)
                # Combine with the causal mask
                sliding_mask = (full_mask + sliding_part).reshape(1, 1, seq_len, seq_len)

        cos = self.cos_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.head_dim).expand(bsz, 1, seq_len, self.head_dim)
        sin = self.sin_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.head_dim).expand(bsz, 1, seq_len, self.head_dim)
        
        new_states_list = []
        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i] if self.config.layer_types else "full_attention"
            attention_mask = sliding_mask if layer_type == "sliding_attention" and sliding_mask is not None else full_mask

            past_st = past_states[i] if past_states else None
            h, new_st = layer(h, attention_mask, past_st, (cos, sin), start_pos, **kwargs)
            new_states_list.append(new_st)
            if i + 1 == len(self.layers): h = self.norm(h)
            if output_hidden_states: all_hidden_states += (h,)

        if output_hidden_states: all_hidden_states += (h,)
        return h, new_states_list, all_hidden_states

class Exaone4ForCausalLM(BaseForCausalLM):
    def _create_model(self, config: BaseConfig, linear_class: Type) -> BaseModel:
        return Exaone4Model(config, linear_class)

    @classmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        return Exaone4Config.from_hf_config(config_dict)

    def _get_key_map(self) -> dict:
        key_map = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i in range(self.config.num_hidden_layers):
            p = f"model.layers.{i}"
            key_map.update({
                f"{p}.post_attention_layernorm.weight": f"{p}.post_attention_layernorm.weight",
                f"{p}.post_feedforward_layernorm.weight": f"{p}.post_feedforward_layernorm.weight",
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