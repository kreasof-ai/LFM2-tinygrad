# src/model/olmoe_modeling.py

import json
import collections
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type

# Third-party imports
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch

# tinygrad imports
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn import Linear, RMSNorm

# Project imports
from model.base_modeling import BaseConfig, BaseModel, BaseForCausalLM
from utils.output import CausalLMOutputWithPast
from utils.rope import apply_rotary_pos_emb

# --- Configuration ---
@dataclass
class OlmoeConfig(BaseConfig):
    hidden_act: str = "silu"
    qk_norm: bool = True
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    # MoE specific
    num_experts: int = 64
    num_experts_per_tok: int = 8
    router_aux_loss_coef: float = 0.01

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "OlmoeConfig":
        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            head_dim=config_dict["hidden_size"] // config_dict["num_attention_heads"],
            rms_norm_eps=config_dict["rms_norm_eps"],
            rope_theta=config_dict["rope_theta"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
            # MoE specific
            num_experts=config_dict["num_experts"],
            num_experts_per_tok=config_dict["num_experts_per_tok"],
            router_aux_loss_coef=config_dict["router_aux_loss_coef"],
        )

# --- Model Components ---

class OlmoeAttention:
    """
    Self-contained attention module for Gemma2. Does not use BaseAttention because it requires different QK norm.
    """
    def __init__(self, config: OlmoeConfig, linear_class: Type = Linear):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = linear_class(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Support for models with QK Norm
        self.q_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps) if getattr(config, "qk_norm") else lambda x: x
        self.k_norm = RMSNorm((self.hidden_size // self.num_heads) * self.num_key_value_heads, eps=config.rms_norm_eps) if getattr(config, "qk_norm") else lambda x: x

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Any], cos_sin: Tuple[Tensor, Tensor], start_pos: int, batch_idx: Optional[Tensor] = None, seq_lens: Optional[List[int]] = None):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        query_states = query_states.permute(0, 2, 1, 3)
        key_states = key_states.permute(0, 2, 1, 3)
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
        else: # Standard KV Caching
            past_key_value = past_kv
            if past_key_value is not None:
                key_states = Tensor.cat(past_key_value[0], key_states, dim=2)
                value_states = Tensor.cat(past_key_value[1], value_states, dim=2)
            all_key_states, all_value_states = key_states, value_states
            present_kv = (key_states, value_states)

        all_key_states = all_key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        all_value_states = all_value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)

        attn_output = Tensor.scaled_dot_product_attention(query_states, all_key_states, all_value_states, attn_mask=attention_mask)
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_kv

class OlmoeMixtureFeedForward:
    def __init__(self, config: OlmoeConfig):
        self.config = config
        self.gate = Linear(config.hidden_size, config.num_experts, bias=False)
        # These are placeholders for the stacked weights from all experts.
        # The weight loader will populate them.
        self.gate_proj = Tensor.zeros(config.num_experts, config.intermediate_size, config.hidden_size)
        self.up_proj = Tensor.zeros(config.num_experts, config.intermediate_size, config.hidden_size)
        self.down_proj = Tensor.zeros(config.num_experts, config.hidden_size, config.intermediate_size)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, dim = x.shape
        T = bsz * seq_len
        K = self.config.num_experts_per_tok
        H = self.config.intermediate_size
        x_flat = x.reshape(T, dim)

        router_logits = self.gate(x_flat) # (T, num_experts)
        router_probs = router_logits.softmax(-1)
        
        top_k_probs, top_k_indices = router_probs.topk(K) # (T, K), (T, K)
        top_k_probs /= top_k_probs.sum(axis=-1, keepdim=True) # Normalize probs

        # Gather expert weights for each token
        W_g = self.gate_proj[top_k_indices] # (T, K, H, D)
        W_u = self.up_proj[top_k_indices]   # (T, K, H, D)
        W_d = self.down_proj[top_k_indices] # (T, K, D, H)

        # Reshape for batched matmul: (T*K) batches of (1,D) @ (D,H)
        x_expanded = x_flat.unsqueeze(1).expand(-1, K, -1).reshape(T * K, 1, dim)
        W_g_reshaped = W_g.transpose(-1, -2).reshape(T * K, dim, H)
        W_u_reshaped = W_u.transpose(-1, -2).reshape(T * K, dim, H)

        # Expert computations
        y_g = x_expanded @ W_g_reshaped # (T*K, 1, H)
        y_u = x_expanded @ W_u_reshaped # (T*K, 1, H)
        y_ff = y_g.silu() * y_u         # (T*K, 1, H)
        
        W_d_reshaped = W_d.transpose(-1, -2).reshape(T * K, H, dim)
        y_out_flat = y_ff @ W_d_reshaped # (T*K, 1, D)

        # Un-reshape and combine with probabilities
        y_out = y_out_flat.reshape(T, K, dim)
        final_y = (y_out * top_k_probs.unsqueeze(-1)).sum(axis=1) # Weighted sum over experts
        
        return final_y.reshape(bsz, seq_len, dim), router_logits

class OlmoeDecoderLayer:
    def __init__(self, config: OlmoeConfig, linear_class: Type):
        self.self_attn = OlmoeAttention(config, linear_class)
        self.mlp = OlmoeMixtureFeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Any], cos_sin: Tuple[Tensor, Tensor], start_pos: int, **kwargs):
        residual = hidden_states
        normed_hidden = self.input_layernorm(hidden_states)
        attn_output, new_kv = self.self_attn(normed_hidden, attention_mask, past_kv, cos_sin, start_pos, **kwargs)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        normed_hidden = self.post_attention_layernorm(hidden_states)
        mlp_output, router_logits = self.mlp(normed_hidden)
        hidden_states = residual + mlp_output
        
        return hidden_states, new_kv, router_logits

class OlmoeModel(BaseModel):
    def _create_decoder_layer(self, config: BaseConfig, linear_class: Type, layer_idx: int):
        return OlmoeDecoderLayer(config, linear_class)
    
    # Override BaseModel's call to handle router_logits
    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int, output_hidden_states: bool, **kwargs):
        h = self.embed_tokens(input_ids)
        bsz, seq_len, _ = h.shape
        all_hidden_states = (h,) if output_hidden_states else None
        mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1).realize() if seq_len > 1 else None
        
        cos = self.cos_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.head_dim).expand(bsz, 1, seq_len, self.head_dim)
        sin = self.sin_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.head_dim).expand(bsz, 1, seq_len, self.head_dim)
        
        all_router_logits = ()
        new_states_list = []
        for i, layer in enumerate(self.layers):
            past_st = past_states[i] if past_states else None
            h, new_st, router_logits = layer(h, mask, past_st, (cos, sin), start_pos, **kwargs)
            all_router_logits += (router_logits,)
            new_states_list.append(new_st)
            if i + 1 == len(self.layers): h = self.norm(h)
            if output_hidden_states: all_hidden_states += (h,)

        if output_hidden_states: all_hidden_states += (h,)
        return h, new_states_list, all_hidden_states, all_router_logits

class OlmoeForCausalLM(BaseForCausalLM):
    def _create_model(self, config: OlmoeConfig, linear_class: Type) -> BaseModel:
        return OlmoeModel(config, linear_class)
    
    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]] = None, start_pos: int = 0, labels: Optional[Tensor] = None, output_hidden_states: bool = False, **kwargs) -> CausalLMOutputWithPast:
        bsz, seq_len = input_ids.shape
        hidden_states, new_states, all_hidden_states, all_router_logits = self.model(
            input_ids, past_states, start_pos, output_hidden_states, **kwargs
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            ce_loss = logits[..., :-1, :].flatten(0, 1).sparse_categorical_crossentropy(labels[..., 1:].flatten(), ignore_index=-100)
            
            # Calculate router auxiliary loss
            total_aux_loss = 0
            for router_logits in all_router_logits: # router_logits is (T, E)
                router_probs = router_logits.softmax(dim=-1)
                # Un-flatten to (B, S, E) to compute mean over batch and sequence
                router_probs_unflat = router_probs.reshape(bsz, seq_len, self.config.num_experts)
                avg_router_probs = router_probs_unflat.mean((0, 1)) # (E,)
                avg_router_logits = router_logits.reshape(bsz, seq_len, self.config.num_experts).mean((0,1))
                
                # Formula from HF implementation for load balancing
                aux_loss_layer = (avg_router_probs * avg_router_logits).sum() * self.config.num_experts
                total_aux_loss += aux_loss_layer
            
            loss = ce_loss + self.config.router_aux_loss_coef * total_aux_loss

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=new_states, hidden_states=all_hidden_states
        )
    
    @classmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        return OlmoeConfig.from_hf_config(config_dict)
    
    def _get_key_map(self) -> dict:
        # NOTE: MoE expert weights are handled by the custom _load_from_hf method.
        # This map is for all non-expert weights.
        key_map = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i in range(self.config.num_hidden_layers):
            p_hf = f"model.layers.{i}"
            p_tg = f"model.layers.{i}"
            key_map.update({
                f"{p_hf}.input_layernorm.weight": f"{p_tg}.input_layernorm.weight",
                f"{p_hf}.post_attention_layernorm.weight": f"{p_tg}.post_attention_layernorm.weight",
                f"{p_hf}.self_attn.q_proj.weight": f"{p_tg}.self_attn.q_proj.weight",
                f"{p_hf}.self_attn.k_proj.weight": f"{p_tg}.self_attn.k_proj.weight",
                f"{p_hf}.self_attn.v_proj.weight": f"{p_tg}.self_attn.v_proj.weight",
                f"{p_hf}.self_attn.o_proj.weight": f"{p_tg}.self_attn.o_proj.weight",
                f"{p_hf}.self_attn.q_norm.weight": f"{p_tg}.self_attn.q_norm.weight",
                f"{p_hf}.self_attn.k_norm.weight": f"{p_tg}.self_attn.k_norm.weight",
                f"{p_hf}.mlp.gate.weight": f"{p_tg}.mlp.gate.weight",
                # The stacked expert weights will have these keys after processing
                f"{p_hf}.mlp.gate_proj.weight": f"{p_tg}.mlp.gate_proj.weight",
                f"{p_hf}.mlp.up_proj.weight": f"{p_tg}.mlp.up_proj.weight",
                f"{p_hf}.mlp.down_proj.weight": f"{p_tg}.mlp.down_proj.weight",
            })
        return key_map

    @classmethod
    def _load_from_hf(cls, model, repo_id: str):
        # This method is overridden to handle the specific format of MoE weights,
        # where individual expert weights are stacked into a single tensor per layer.
        print("Using custom OLMoE weight loader...")
        
        hf_state_dict = {}
        # 1. Load all weights from all shards into a single dict in memory
        try:
            index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
            with open(index_path) as f: weight_map = json.load(f)["weight_map"]
            
            shards = set(weight_map.values())
            print(f"Model is sharded. Found {len(shards)} files.")
            for shard_filename in sorted(list(shards)):
                print(f"  Loading shard: {shard_filename}")
                local_shard_path = hf_hub_download(repo_id=repo_id, filename=shard_filename)
                with safe_open(local_shard_path, framework="pt", device="cpu") as f:
                    for hf_key in f.keys():
                        hf_state_dict[hf_key] = f.get_tensor(hf_key)
        except Exception:
            print("No index found, falling back to single 'model.safetensors' file.")
            local_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            with safe_open(local_path, framework="pt", device="cpu") as f:
                for hf_key in f.keys():
                    hf_state_dict[hf_key] = f.get_tensor(hf_key)
        
        # 2. Process MoE weights: group, stack, and replace
        expert_groups = collections.defaultdict(lambda: collections.defaultdict(dict))
        keys_to_delete = []
        
        for hf_key in hf_state_dict.keys():
            if ".mlp.experts." in hf_key:
                parts = hf_key.split('.')
                layer_idx, expert_idx, param_name = parts[2], int(parts[5]), parts[6]
                expert_groups[layer_idx][param_name][expert_idx] = hf_state_dict[hf_key]
                keys_to_delete.append(hf_key)
        
        for k in keys_to_delete: del hf_state_dict[k]
        
        for layer_idx, params in expert_groups.items():
            for param_name, experts in params.items():
                stacked_tensor = torch.stack([experts[i] for i in sorted(experts.keys())])
                new_key = f"model.layers.{layer_idx}.mlp.{param_name}.weight"
                hf_state_dict[new_key] = stacked_tensor
                print(f"  Stacked MoE weights for {new_key} with shape {stacked_tensor.shape}")

        # 3. Map processed HF keys to tinygrad keys and create tinygrad state dict
        tg_state_dict = {}
        key_map = model._get_key_map()
        for hf_key, tg_key in key_map.items():
            if hf_key in hf_state_dict:
                tensor = hf_state_dict[hf_key]
                # Permute QK weights for RoPE, as HF stores them differently
                if "self_attn.q_proj" in hf_key:
                    tensor = tensor.reshape(model.config.num_attention_heads, 2, -1, tensor.shape[-1]).transpose(1, 2).reshape(*tensor.shape)
                if "self_attn.k_proj" in hf_key:
                    tensor = tensor.reshape(model.config.num_key_value_heads, 2, -1, tensor.shape[-1]).transpose(1, 2).reshape(*tensor.shape)

                tg_state_dict[tg_key] = Tensor(tensor.to(torch.float32).numpy(), requires_grad=False, device=Device.DEFAULT)

        # 4. Finalize and load into model (replicated from BaseForCausalLM)
        if model.config.quantize: raise NotImplementedError("Quantization is not supported for OLMoE models yet.")
        
        for k in tg_state_dict:
            if tg_state_dict[k].dtype != dtypes.uint8:
                tg_state_dict[k] = tg_state_dict[k].cast(model.config.dtype)
        
        from tinygrad.nn.state import load_state_dict
        print("Assigning weights to model...")
        load_state_dict(model, tg_state_dict, strict=False)
        print("All weights loaded and assigned.")