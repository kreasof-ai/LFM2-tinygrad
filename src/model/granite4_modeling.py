# src/model/granite4_modeling.py

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type

# Third-party imports
from huggingface_hub import hf_hub_download

# tinygrad imports
from tinygrad import Tensor, dtypes
from tinygrad.nn import Conv1d, Linear, RMSNorm, Embedding

# Project imports
from model.base_modeling import BaseConfig, BaseForCausalLM, BaseModel

# --- Configuration ---
@dataclass
class Granite4Config(BaseConfig):
    attention_bias: bool = False
    hidden_act: str = "silu"
    
    # Granite-4 specific
    attention_multiplier: float = 0.015625
    embedding_multiplier: float = 12.0
    logits_scaling: float = 3.0
    residual_multiplier: float = 0.246
    layer_types: List[str] = field(default_factory=list)
    shared_intermediate_size: int = 2048
    num_local_experts: int = 0
    num_experts_per_tok: int = 0

    # Mamba specific
    mamba_chunk_size: int = 256
    mamba_conv_bias: bool = True
    mamba_d_conv: int = 4
    mamba_d_head: int = 32
    mamba_d_state: int = 128
    mamba_expand: int = 2
    mamba_n_groups: int = 1
    mamba_n_heads: int = 48
    mamba_proj_bias: bool = False
    
    # Not used by our implementation but part of config
    position_embedding_type: str = "nope"

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "Granite4Config":
        head_dim = config_dict["hidden_size"] // config_dict["num_attention_heads"]
        return cls(
            vocab_size=config_dict["vocab_size"], hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"], num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"], num_key_value_heads=config_dict["num_key_value_heads"],
            head_dim=head_dim, rms_norm_eps=config_dict["rms_norm_eps"], rope_theta=config_dict["rope_theta"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
            # Granite-4 specific
            attention_multiplier=config_dict["attention_multiplier"], embedding_multiplier=config_dict["embedding_multiplier"],
            logits_scaling=config_dict["logits_scaling"], residual_multiplier=config_dict["residual_multiplier"],
            layer_types=config_dict["layer_types"], shared_intermediate_size=config_dict["shared_intermediate_size"],
            num_local_experts=config_dict["num_local_experts"], num_experts_per_tok=config_dict["num_experts_per_tok"],
            # Mamba specific
            mamba_chunk_size=config_dict["mamba_chunk_size"], mamba_conv_bias=config_dict["mamba_conv_bias"],
            mamba_d_conv=config_dict["mamba_d_conv"], mamba_d_head=config_dict["mamba_d_head"],
            mamba_d_state=config_dict["mamba_d_state"], mamba_expand=config_dict["mamba_expand"],
            mamba_n_groups=config_dict["mamba_n_groups"], mamba_n_heads=config_dict["mamba_n_heads"],
            mamba_proj_bias=config_dict["mamba_proj_bias"],
        )

# --- Model Components ---

class Granite4RMSNormGated:
    """ Custom RMSNorm for Mamba with (x * silu(gate)) input. """
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x: Tensor, gate: Tensor) -> Tensor:
        x_gated = x * gate.silu()
        normed_x = x_gated * (x_gated.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return normed_x * self.weight

class Granite4SharedMLP:
    """ Gated MLP used in the dense part of each decoder layer. """
    def __init__(self, config: Granite4Config, linear_class: Type = Linear):
        self.input_linear = linear_class(config.hidden_size, config.shared_intermediate_size * 2, bias=False)
        self.output_linear = linear_class(config.shared_intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        gate, up = self.input_linear(x).chunk(2, dim=-1)
        return self.output_linear(gate.silu() * up)

class Granite4Attention:
    """ Attention module with Granite-4's specific scaling factor. """
    def __init__(self, config: Granite4Config, linear_class: Type = Linear):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.attention_multiplier

        self.q_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = linear_class(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Any]):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key_states = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)
        value_states = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)

        if past_kv is not None:
            key_states = Tensor.cat(past_kv[0], key_states, dim=2)
            value_states = Tensor.cat(past_kv[1], value_states, dim=2)
        present_kv = (key_states, value_states)

        key_states = key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        value_states = value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)

        attn_weights = (query_states @ key_states.transpose(-2, -1)) * self.scaling
        if attention_mask is not None: attn_weights += attention_mask
        attn_weights = attn_weights.softmax(-1)
        attn_output = attn_weights @ value_states
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_kv

class Granite4MambaLayer:
    """ tinygrad implementation of the Mamba state space model layer. """
    def __init__(self, config: Granite4Config, linear_class: Type = Linear):
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = int(config.mamba_expand * config.hidden_size)
        self.num_heads = config.mamba_n_heads
        self.head_dim = config.mamba_d_head
        self.n_groups = config.mamba_n_groups
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = linear_class(self.hidden_size, projection_size, bias=config.mamba_proj_bias)
        self.conv1d = Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=config.mamba_conv_bias,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1
        )
        self.out_proj = linear_class(self.intermediate_size, self.hidden_size, bias=config.mamba_proj_bias)

        self.dt_bias = Tensor.ones(self.num_heads, requires_grad=True)
        self.A_log = Tensor.arange(1, self.num_heads + 1).log().requires_grad_()
        self.D = Tensor.ones(self.num_heads, requires_grad=True)
        self.norm = Granite4RMSNormGated(self.intermediate_size, eps=config.rms_norm_eps)

    def __call__(self, x: Tensor, past_state: Optional[Tuple[Tensor, Tensor]]):
        bsz, seq_len, _ = x.shape
        
        proj_out = self.in_proj(x)
        gate, x_bc, dt = proj_out.split([self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
        
        # 1D Conv
        x_bc_permuted = x_bc.permute(0, 2, 1)
        if seq_len > 1:
            conv_out = self.conv1d(x_bc_permuted)[:, :, :seq_len]
            new_conv_state = x_bc_permuted[:, :, -(self.conv_kernel_size-1):].pad(((0,0),(0,0),(self.conv_kernel_size-1-x_bc_permuted.shape[2],0))) if x_bc_permuted.shape[2] < self.conv_kernel_size-1 else x_bc_permuted[:,:,-self.conv_kernel_size+1:]
        else: # Generation
            past_conv_state = past_state[0] if past_state is not None else Tensor.zeros(bsz, self.conv_dim, self.conv_kernel_size - 1, dtype=x.dtype)
            conv_in = Tensor.cat(past_conv_state, x_bc_permuted, dim=2)
            conv_out = self.conv1d(conv_in)[:, :, self.conv_kernel_size - 1:self.conv_kernel_size]
            new_conv_state = conv_in[:, :, 1:]

        conv_out = conv_out.permute(0, 2, 1).silu()
        hidden, B, C = conv_out.split([self.intermediate_size, self.n_groups*self.ssm_state_size, self.n_groups*self.ssm_state_size], dim=-1)
        
        # SSM Recurrence
        A = -self.A_log.exp()
        D = self.D
        dt = (dt + self.dt_bias).softplus()
        
        ssm_state = past_state[1] if past_state is not None else Tensor.zeros(bsz, self.num_heads, self.head_dim, self.ssm_state_size, dtype=x.dtype)
        
        # This is a simplified recurrent implementation for clarity and correctness.
        # A full parallel scan (SSD) would be more performant for prefill but is highly complex.
        outputs = []
        for i in range(seq_len):
            dt_i = dt[:, i, :]
            hidden_i = hidden[:, i, :].reshape(bsz, self.num_heads, self.head_dim)
            B_i = B[:, i, :].reshape(bsz, self.n_groups, -1).repeat((1, self.num_heads // self.n_groups, 1)).reshape(bsz, self.num_heads, self.ssm_state_size)
            C_i = C[:, i, :].reshape(bsz, self.n_groups, -1).repeat((1, self.num_heads // self.n_groups, 1)).reshape(bsz, self.num_heads, self.ssm_state_size)

            dA = (dt_i.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-1)).exp()
            dB = dt_i.unsqueeze(-1) * B_i
            dBx = dB.unsqueeze(2) * hidden_i.unsqueeze(3)
            
            ssm_state = ssm_state * dA.unsqueeze(2) + dBx
            y_i = (ssm_state * C_i.unsqueeze(2)).sum(-1) + hidden_i * D.unsqueeze(0).unsqueeze(-1)
            outputs.append(y_i.reshape(bsz, self.intermediate_size))
        
        y = Tensor.stack(*outputs, dim=1) if outputs else Tensor.zeros(bsz, 0, self.intermediate_size, dtype=x.dtype)
        new_ssm_state = ssm_state
        
        return self.out_proj(self.norm(y, gate)), (new_conv_state.contiguous(), new_ssm_state.contiguous())

class Granite4DecoderLayer:
    def __init__(self, config: Granite4Config, layer_idx: int, linear_class: Type):

        self.layer_type = config.layer_types[layer_idx]
        self.residual_multiplier = config.residual_multiplier

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if config.num_local_experts > 0:
            raise NotImplementedError("Mixture of Experts is not yet supported in OpenFormer for Granite-4.")
        self.mlp = Granite4SharedMLP(config, linear_class)
        
        if self.layer_type == "attention": self.op = Granite4Attention(config, linear_class)
        elif self.layer_type == "mamba": self.op = Granite4MambaLayer(config, linear_class)
        else: raise ValueError(f"Unknown layer type: {self.layer_type}")

    def __call__(self, x: Tensor, attention_mask: Optional[Tensor], past_state: Optional[Any]):
        residual = x
        x_norm = self.input_layernorm(x)

        if self.layer_type == "attention":
            op_out, new_state = self.op(x_norm, attention_mask, past_state)
        else:
            op_out, new_state = self.op(x_norm, past_state)
        
        x = residual + op_out * self.residual_multiplier
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x * self.residual_multiplier
        return x, new_state

class Granite4Model(BaseModel):
    def __init__(self, config: Granite4Config, linear_class: Type):
        super().__init__(config, linear_class) # This will call _create_decoder_layer
        self.embedding_multiplier = config.embedding_multiplier

    def _create_decoder_layer(self, config: Granite4Config, linear_class: Type, layer_idx: int):
        return Granite4DecoderLayer(config, layer_idx, linear_class)
    
    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int, output_hidden_states: bool, **kwargs):
        h = self.embed_tokens(input_ids) * self.embedding_multiplier
        bsz, seq_len, _ = h.shape
        all_hidden_states = (h,) if output_hidden_states else None
        
        mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1).realize() if seq_len > 1 else None
        
        new_states_list = []
        for i, layer in enumerate(self.layers):
            past_st = past_states[i] if past_states else None
            h, new_st = layer(h, mask, past_st, **kwargs)
            new_states_list.append(new_st)
            if i + 1 == len(self.layers): h = self.norm(h)
            if output_hidden_states: all_hidden_states += (h,)
        
        return h, new_states_list, all_hidden_states


class Granite4ForCausalLM(BaseForCausalLM):
    def __init__(self, config: Granite4Config):
        if config.num_local_experts > 0:
            raise NotImplementedError("Mixture of Experts is not yet supported for Granite-4.")
        super().__init__(config)
        self.logits_scaling = config.logits_scaling

    def _create_model(self, config: BaseConfig, linear_class: Type) -> BaseModel:
        return Granite4Model(config, linear_class)

    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        output.logits = output.logits / self.logits_scaling
        return output

    @classmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        return Granite4Config.from_hf_config(config_dict)
    
    def _get_key_map(self) -> dict:
        key_map = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight", "lm_head.weight": "lm_head.weight",
        }
        for i, layer_type in enumerate(self.config.layer_types):
            p = f"model.layers.{i}"
            key_map.update({
                f"{p}.input_layernorm.weight": f"{p}.input_layernorm.weight",
                f"{p}.post_attention_layernorm.weight": f"{p}.post_attention_layernorm.weight",
                f"{p}.shared_mlp.input_linear.weight": f"{p}.mlp.input_linear.weight",
                f"{p}.shared_mlp.output_linear.weight": f"{p}.mlp.output_linear.weight",
            })
            if layer_type == "attention":
                op_p = f"{p}.op"
                key_map.update({
                    f"{p}.self_attn.q_proj.weight": f"{op_p}.q_proj.weight",
                    f"{p}.self_attn.k_proj.weight": f"{op_p}.k_proj.weight",
                    f"{p}.self_attn.v_proj.weight": f"{op_p}.v_proj.weight",
                    f"{p}.self_attn.o_proj.weight": f"{op_p}.o_proj.weight",
                })
            else: # mamba
                op_p = f"{p}.op"
                key_map.update({
                    f"{p}.mamba.in_proj.weight": f"{op_p}.in_proj.weight",
                    f"{p}.mamba.conv1d.weight": f"{op_p}.conv1d.weight",
                    f"{p}.mamba.conv1d.bias": f"{op_p}.conv1d.bias",
                    f"{p}.mamba.dt_bias": f"{op_p}.dt_bias",
                    f"{p}.mamba.A_log": f"{op_p}.A_log",
                    f"{p}.mamba.D": f"{op_p}.D",
                    f"{p}.mamba.out_proj.weight": f"{op_p}.out_proj.weight",
                    f"{p}.mamba.norm.weight": f"{op_p}.norm.weight",
                })
        return key_map