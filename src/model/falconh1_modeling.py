# src/model/falconh1_modeling.py

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
from utils.rope import apply_rotary_pos_emb

# --- Configuration ---
@dataclass
class FalconH1Config(BaseConfig):
    hidden_act: str = "silu"
    attention_bias: bool = False
    
    # Falcon-H1 specific
    embedding_multiplier: float = 5.65
    lm_head_multiplier: float = 0.039
    key_multiplier: float = 0.39
    projectors_bias: bool = False
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 0.9375
    ssm_in_multiplier: float = 1.25
    ssm_out_multiplier: float = 0.2357
    mlp_multipliers: List[float] = field(default_factory=list)
    ssm_multipliers: List[float] = field(default_factory=list)
    
    # Mamba specific
    mamba_chunk_size: int = 128
    mamba_conv_bias: bool = True
    mamba_d_conv: int = 4
    mamba_d_head: int = 64
    mamba_d_ssm: int = 1536
    mamba_d_state: int = 128
    mamba_expand: int = 2
    mamba_n_groups: int = 1
    mamba_n_heads: int = 24
    mamba_proj_bias: bool = False

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "FalconH1Config":
        return cls(
            vocab_size=config_dict["vocab_size"], hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict["intermediate_size"], num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"], num_key_value_heads=config_dict["num_key_value_heads"],
            head_dim=config_dict["head_dim"], rms_norm_eps=config_dict["rms_norm_eps"], rope_theta=config_dict["rope_theta"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
            # Falcon-H1 specific
            embedding_multiplier=config_dict["embedding_multiplier"], lm_head_multiplier=config_dict["lm_head_multiplier"],
            key_multiplier=config_dict["key_multiplier"], attention_in_multiplier=config_dict["attention_in_multiplier"],
            projectors_bias=config_dict["projectors_bias"],
            attention_out_multiplier=config_dict["attention_out_multiplier"], ssm_in_multiplier=config_dict["ssm_in_multiplier"],
            ssm_out_multiplier=config_dict["ssm_out_multiplier"], mlp_multipliers=config_dict["mlp_multipliers"],
            ssm_multipliers=config_dict["ssm_multipliers"],
            # Mamba specific
            mamba_chunk_size=config_dict["mamba_chunk_size"], mamba_conv_bias=config_dict["mamba_conv_bias"],
            mamba_d_conv=config_dict["mamba_d_conv"], mamba_d_head=config_dict["mamba_d_head"],
            mamba_d_ssm=config_dict["mamba_d_ssm"], mamba_d_state=config_dict["mamba_d_state"],
            mamba_expand=config_dict["mamba_expand"], mamba_n_groups=config_dict["mamba_n_groups"],
            mamba_n_heads=config_dict["mamba_n_heads"], mamba_proj_bias=config_dict["mamba_proj_bias"],
        )

# --- Model Components ---

class FalconH1MLP:
    def __init__(self, config: FalconH1Config, linear_class: Type = Linear):
        self.gate_proj = linear_class(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers

    def __call__(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x) * self.gate_multiplier
        up = self.up_proj(x)
        y = self.down_proj(up * gate.silu()) * self.down_multiplier
        return y

class FalconH1Attention:
    def __init__(self, config: FalconH1Config, linear_class: Type = Linear):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.key_multiplier = config.key_multiplier
        self.scaling = self.head_dim ** -0.5

        self.q_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = linear_class(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def __call__(self, x: Tensor, mask: Optional[Tensor], past_kv: Optional[Any], cos_sin: Tuple[Tensor, Tensor]):
        bsz, q_len, _ = x.shape
        query_states = self.q_proj(x).reshape(bsz, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key_states = (self.k_proj(x) * self.key_multiplier).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)
        value_states = self.v_proj(x).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_sin[0], cos_sin[1])

        if past_kv:
            key_states = Tensor.cat(past_kv[0], key_states, dim=2)
            value_states = Tensor.cat(past_kv[1], value_states, dim=2)
        
        present_kv = (key_states, value_states)

        key_states_grouped = key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        value_states_grouped = value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        
        attn_output = Tensor.scaled_dot_product_attention(query_states, key_states_grouped, value_states_grouped, attn_mask=mask)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), present_kv

class FalconH1MambaLayer:
    def __init__(self, config: FalconH1Config, linear_class: Type = Linear):
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.intermediate_size = config.mamba_d_ssm
        self.num_heads = config.mamba_n_heads
        self.head_dim = config.mamba_d_head
        self.n_groups = config.mamba_n_groups
        self.conv_kernel_size = config.mamba_d_conv
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = linear_class(self.hidden_size, projection_size, bias=config.mamba_proj_bias)
        self.conv1d = Conv1d(self.conv_dim, self.conv_dim, self.conv_kernel_size, groups=self.conv_dim, bias=config.mamba_conv_bias, padding=self.conv_kernel_size - 1)
        self.out_proj = linear_class(self.intermediate_size, config.hidden_size, bias=config.projectors_bias)
        
        # Construct MuP vector for SSM multipliers
        z_mult, x_mult, b_mult, c_mult, dt_mult = config.ssm_multipliers
        groups_state_size = self.n_groups * self.ssm_state_size
        mup_vector = Tensor.cat(
            Tensor.full((self.intermediate_size,), z_mult),       # gate
            Tensor.full((self.intermediate_size,), x_mult),       # x
            Tensor.full((groups_state_size,), b_mult),          # B
            Tensor.full((groups_state_size,), c_mult),          # C
            Tensor.full((self.num_heads,), dt_mult)            # dt
        ).reshape(1, 1, -1)
        self.mup_vector = mup_vector

        self.dt_bias = Tensor.ones(self.num_heads)
        self.A_log = Tensor.arange(1, self.num_heads + 1).log()
        self.D = Tensor.ones(self.num_heads)
        self.ssm_in_multiplier = config.ssm_in_multiplier

    def __call__(self, x: Tensor, past_state: Optional[Tuple[Tensor, Tensor]]):
        bsz, seq_len, _ = x.shape
        
        proj_out = self.in_proj(x * self.ssm_in_multiplier) * self.mup_vector
        gate, x_bc, dt = proj_out.split([self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
        
        x_bc_permuted = x_bc.permute(0, 2, 1)
        if seq_len > 1:
            conv_out = self.conv1d(x_bc_permuted)[:, :, :seq_len]
            new_conv_state = x_bc_permuted[:, :, -(self.conv_kernel_size - 1):].pad(((0,0),(0,0),(self.conv_kernel_size-1-x_bc_permuted.shape[2],0))) if x_bc_permuted.shape[2] < self.conv_kernel_size-1 else x_bc_permuted[:,:,-self.conv_kernel_size+1:]
        else:
            past_conv_state = past_state[0] if past_state else Tensor.zeros(bsz, self.conv_dim, self.conv_kernel_size - 1, dtype=x.dtype)
            conv_in = Tensor.cat(past_conv_state, x_bc_permuted, dim=2)
            conv_out = self.conv1d(conv_in)[:, :, self.conv_kernel_size-1:self.conv_kernel_size]
            new_conv_state = conv_in[:, :, 1:]
        
        conv_out = conv_out.permute(0, 2, 1).silu()
        x_ssm, B, C = conv_out.split([self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups*self.ssm_state_size], dim=-1)

        A = -self.A_log.exp()
        dt = (dt + self.dt_bias).softplus()
        ssm_state = past_state[1] if past_state else Tensor.zeros(bsz, self.num_heads, self.head_dim, self.ssm_state_size, dtype=x.dtype)

        outputs = []
        for i in range(seq_len):
            dt_i = dt[:, i, :]
            x_ssm_i = x_ssm[:, i, :].reshape(bsz, self.num_heads, self.head_dim)
            B_i = B[:, i, :].reshape(bsz, self.n_groups, -1).repeat((1, self.num_heads // self.n_groups, 1)).reshape(bsz, self.num_heads, self.ssm_state_size)
            C_i = C[:, i, :].reshape(bsz, self.n_groups, -1).repeat((1, self.num_heads // self.n_groups, 1)).reshape(bsz, self.num_heads, self.ssm_state_size)

            dA = (dt_i.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-1)).exp()
            dB = dt_i.unsqueeze(-1) * B_i
            dBx = dB.unsqueeze(2) * x_ssm_i.unsqueeze(3)
            
            ssm_state = ssm_state * dA.unsqueeze(2) + dBx
            y_i = (ssm_state * C_i.unsqueeze(2)).sum(-1) + x_ssm_i * self.D.unsqueeze(0).unsqueeze(-1)
            outputs.append(y_i.reshape(bsz, self.intermediate_size))
        
        y = Tensor.stack(*outputs, dim=1) if outputs else Tensor.zeros(bsz, 0, self.intermediate_size, dtype=x.dtype)
        
        return self.out_proj(y * gate.silu()), (new_conv_state.contiguous(), ssm_state.contiguous())

class FalconH1DecoderLayer:
    def __init__(self, config: FalconH1Config, layer_idx: int, linear_class: Type):
        self.mamba = FalconH1MambaLayer(config, linear_class)
        self.self_attn = FalconH1Attention(config, linear_class)
        self.feed_forward = FalconH1MLP(config, linear_class)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.attention_in_multiplier = config.attention_in_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier

    def __call__(self, x: Tensor, mask: Optional[Tensor], past_state: Optional[Any], cos_sin: Tuple[Tensor, Tensor]):
        residual = x
        x_norm = self.input_layernorm(x)

        mamba_state = past_state[0] if past_state else None
        attn_state = past_state[1] if past_state else None
        
        mamba_out, new_mamba_state = self.mamba(x_norm, mamba_state)
        attn_out, new_attn_state = self.self_attn(x_norm * self.attention_in_multiplier, mask, attn_state, cos_sin)

        x = residual + (mamba_out * self.ssm_out_multiplier) + (attn_out * self.attn_out_multiplier)
        
        residual = x
        x_norm_ff = self.pre_ff_layernorm(x)
        x = residual + self.feed_forward(x_norm_ff)
        
        return x, (new_mamba_state, new_attn_state)

class FalconH1Model(BaseModel):
    def __init__(self, config: FalconH1Config, linear_class: Type):
        super().__init__(config, linear_class)
        self.embedding_multiplier = config.embedding_multiplier

    def _create_decoder_layer(self, config: FalconH1Config, linear_class: Type, layer_idx: int):
        return FalconH1DecoderLayer(config, layer_idx, linear_class)
    
    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int, output_hidden_states: bool, **kwargs):
        h = self.embed_tokens(input_ids) * self.embedding_multiplier
        bsz, seq_len, _ = h.shape
        all_hidden_states = (h,) if output_hidden_states else None
        
        mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1).realize() if seq_len > 1 else None
        cos = self.cos_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.head_dim).expand(bsz, 1, seq_len, self.head_dim)
        sin = self.sin_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.head_dim).expand(bsz, 1, seq_len, self.head_dim)
        
        new_states_list = []
        for i, layer in enumerate(self.layers):
            past_st = past_states[i] if past_states else None
            # The base class calls the layer with a different signature, so we adapt here
            h, new_st = layer(h, mask, past_st, (cos, sin))
            new_states_list.append(new_st)
            
        h = self.norm(h)
        if output_hidden_states: all_hidden_states += (h,)
        return h, new_states_list, all_hidden_states

class FalconH1ForCausalLM(BaseForCausalLM):
    def __init__(self, config: FalconH1Config):
        super().__init__(config)
        self.lm_head_multiplier = config.lm_head_multiplier
        # Rename final_layernorm from HF to norm to match BaseModel
        self.model.norm = self.model.final_layernorm

    def _create_model(self, config: BaseConfig, linear_class: Type) -> BaseModel:
        # BaseModel's __init__ calls _create_decoder_layer, which we need to override.
        # But we need to rename `norm` for HF weight compatibility first.
        model = FalconH1Model(config, linear_class)
        model.final_layernorm = model.norm
        return model

    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        output.logits *= self.lm_head_multiplier
        if output.loss is not None:
             # Recompute loss with scaled logits
             logits, labels = output.logits, kwargs.get("labels")
             output.loss = logits[..., :-1, :].flatten(0, 1).sparse_categorical_crossentropy(labels[..., 1:].flatten(), ignore_index=-100)
        return output

    @classmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        return FalconH1Config.from_hf_config(config_dict)
    
    def _get_key_map(self) -> dict:
        key_map = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.final_layernorm.weight": "model.norm.weight", 
            "lm_head.weight": "lm_head.weight",
        }
        for i in range(self.config.num_hidden_layers):
            p = f"model.layers.{i}"
            key_map.update({
                f"{p}.input_layernorm.weight": f"{p}.input_layernorm.weight",
                f"{p}.pre_ff_layernorm.weight": f"{p}.pre_ff_layernorm.weight",
                f"{p}.feed_forward.gate_proj.weight": f"{p}.feed_forward.gate_proj.weight",
                f"{p}.feed_forward.up_proj.weight": f"{p}.feed_forward.up_proj.weight",
                f"{p}.feed_forward.down_proj.weight": f"{p}.feed_forward.down_proj.weight",
                f"{p}.self_attn.q_proj.weight": f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight": f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight": f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight": f"{p}.self_attn.o_proj.weight",
                f"{p}.mamba.in_proj.weight": f"{p}.mamba.in_proj.weight",
                f"{p}.mamba.conv1d.weight": f"{p}.mamba.conv1d.weight",
                f"{p}.mamba.conv1d.bias": f"{p}.mamba.conv1d.bias",
                f"{p}.mamba.dt_bias": f"{p}.mamba.dt_bias",
                f"{p}.mamba.A_log": f"{p}.mamba.A_log",
                f"{p}.mamba.D": f"{p}.mamba.D",
                f"{p}.mamba.out_proj.weight": f"{p}.mamba.out_proj.weight",
            })
        return key_map