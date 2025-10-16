# src/model/qwen3_modeling.py

"""
Qwen3 tinygrad Implementation

This is a from-scratch implementation of the Qwen3 architecture. It supports standard
float32/float16 inference and quantization.
"""

import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type

# Third-party imports
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch  # only used for converting weights
from transformers import AutoTokenizer

# tinygrad imports
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn import Embedding, Linear, RMSNorm
from tinygrad.nn.state import load_state_dict

# Project imports for shared components
from extra.quantization import NF4Linear, Int8Linear
from utils.rope import _precompute_rope_cache, apply_rotary_pos_emb
from utils.output import CausalLMOutputWithPast

# --- Configuration ---

@dataclass
class Qwen3Config:
    """Configuration class for Qwen3."""
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"

    # --- tinygrad specific flags ---
    dtype: Any = dtypes.float32
    quantize: Optional[str] = None

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "Qwen3Config":
        """Creates a config instance from a Hugging Face config dictionary."""
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

class Qwen3MLP:
    def __init__(self, config: Qwen3Config, linear_class: Type = Linear):
        self.gate_proj = linear_class(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size, bias=False)
    def __call__(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))

class Qwen3Attention:
    def __init__(self, config: Qwen3Config, linear_class: Type = Linear):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = linear_class(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Tuple[Tensor, Tensor]], cos_sin: Tuple[Tensor, Tensor]):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states).permute(0, 2, 1, 3)
        key_states = self.k_norm(key_states).permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)

        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_kv is not None:
            key_states = Tensor.cat(past_kv[0], key_states, dim=2)
            value_states = Tensor.cat(past_kv[1], value_states, dim=2)
        present_kv = (key_states, value_states)

        key_states = key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        value_states = value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        
        attn_output = Tensor.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask)
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_kv

class Qwen3DecoderLayer:
    def __init__(self, config: Qwen3Config, linear_class: Type = Linear):
        self.self_attn = Qwen3Attention(config, linear_class)
        self.mlp = Qwen3MLP(config, linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Tuple[Tensor, Tensor]], cos_sin: Tuple[Tensor, Tensor]):
        residual = hidden_states
        normed_hidden = self.input_layernorm(hidden_states)
        attn_output, new_kv = self.self_attn(normed_hidden, attention_mask, past_kv, cos_sin)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        normed_hidden = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden)
        hidden_states = residual + mlp_output
        return hidden_states, new_kv

class Qwen3Model:
    def __init__(self, config: Qwen3Config, linear_class: Type = Linear):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3DecoderLayer(config, linear_class) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        cos_cache, sin_cache = _precompute_rope_cache(dim=config.head_dim, max_seq_len=config.max_position_embeddings, base=config.rope_theta, dtype=config.dtype)
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int, output_hidden_states: bool):
        h = self.embed_tokens(input_ids)
        bsz, seq_len, _ = h.shape
        all_hidden_states = (h,) if output_hidden_states else None
        mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1).realize() if seq_len > 1 else None
        
        head_dim = self.cos_cache.shape[-1]
        cos = self.cos_cache[start_pos : start_pos + seq_len].reshape(seq_len, head_dim)
        sin = self.sin_cache[start_pos : start_pos + seq_len].reshape(seq_len, head_dim)
        
        new_states_list = []
        for i, layer in enumerate(self.layers):
            past_st = past_states[i] if past_states else None
            h, new_st = layer(h, mask, past_st, (cos, sin))
            new_states_list.append(new_st)
            if output_hidden_states:
                all_hidden_states += (h,)
        
        h = self.norm(h)
        if output_hidden_states: all_hidden_states += (h,)
        
        return h, new_states_list, all_hidden_states

class Qwen3ForCausalLM:
    def __init__(self, config: Qwen3Config):
        self.config = config
        self.tokenizer = None
        
        if config.quantize == "nf4": self.linear_class = NF4Linear()
        elif config.quantize == "int8": self.linear_class = Int8Linear()
        else: self.linear_class = Linear

        self.model = Qwen3Model(config, self.linear_class)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]] = None, start_pos: int = 0, labels: Optional[Tensor] = None, output_hidden_states: bool = False) -> CausalLMOutputWithPast:
        hidden_states, new_states, all_hidden_states = self.model(input_ids, past_states, start_pos, output_hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = logits[..., :-1, :].flatten(0, 1).sparse_categorical_crossentropy(labels[..., 1:].flatten(), ignore_index=-100)

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=new_states, hidden_states=all_hidden_states
        )
    
    def _sample(self, logits: Tensor, do_sample: bool, temperature: float, min_p: float, repetition_penalty: float, generated_tokens: List[int]) -> int:
        """Helper function for token sampling."""
        if repetition_penalty != 1.0 and generated_tokens:
            unique_tokens = Tensor(list(set(generated_tokens)), dtype=dtypes.int32)
            updates = Tensor.where(logits[unique_tokens] > 0, logits[unique_tokens] / repetition_penalty, logits[unique_tokens] * repetition_penalty)
            logits = logits.scatter(0, unique_tokens, updates)

        if not do_sample or temperature == 0:
            return logits.argmax().item()
        
        if min_p > 0.0:
            raise NotImplementedError("Top-p (min_p) sampling is not yet supported in tinygrad due to the lack of a sort operation.")

        probs = (logits / temperature).softmax()
        return (probs.cumsum() > Tensor.uniform(1).item()).argmax().item()
    
    def _decode_one_token(self, next_token_id: int):
        print(self.tokenizer.decode([next_token_id]), end="", flush=True)

    def generate(self, input_ids: Tensor, max_new_tokens: int, do_sample: bool = False, temperature: float = 1.0, min_p: float = 0.0, repetition_penalty: float = 1.0) -> Tensor:
        assert self.tokenizer is not None, "Tokenizer must be attached to the model."
        Tensor.training = False
        
        prompt_len = input_ids.shape[1]
        tokens = input_ids[0].numpy().tolist()

        past_states = [None] * len(self.model.layers)
        outputs = self(Tensor([tokens]), past_states, start_pos=0)
        start_pos = len(tokens)
        
        for _ in range(max_new_tokens):
            past_states = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            next_token = self._sample(logits, do_sample, temperature, min_p, repetition_penalty, tokens)
            tokens.append(next_token)
            if next_token == self.tokenizer.eos_token_id: break

            self._decode_one_token(next_token)

            outputs = self(Tensor([[next_token]]), past_states, start_pos=start_pos)
            start_pos += 1

        return Tensor([tokens], dtype=dtypes.int32)
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        print(f"--- Loading Qwen3 Model: {model_id} ---")
        
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        config = Qwen3Config.from_hf_config(config_dict)

        torch_dtype_map = {"bfloat16": dtypes.bfloat16, "float16": dtypes.float16, "float32": dtypes.float32}
        if "torch_dtype" in kwargs: config.dtype = torch_dtype_map.get(str(kwargs["torch_dtype"]).split('.')[-1], dtypes.float32)
        if "quantize" in kwargs: config.quantize = kwargs["quantize"]

        print("\nInitializing model architecture...")
        model = cls(config)

        load_from_hf(model, model_id)

        print("\nLoading tokenizer...")
        model.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        for k in ["device_map", "attn_implementation"]:
            if k in kwargs: print(f"  [Warning] tinygrad: Argument '{k}' is not used.")

        return model

def load_from_hf(model: Qwen3ForCausalLM, repo_id: str, filename: str = "model.safetensors"):
    print(f"Fetching weights from {repo_id}/{filename}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    dtype = model.config.dtype

    key_map = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for i in range(model.config.num_hidden_layers):
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

    tg_state_dict = {}
    print("Loading weights into memory...")
    with safe_open(local_path, framework="pt", device="cpu") as f:
        for hf_key, tg_key in key_map.items():
            if hf_key not in f.keys():
                print(f"Warning: Weight key not found: {hf_key}")
                continue
            pt_tensor = f.get_tensor(hf_key)
            np_array = pt_tensor.to(torch.float32).numpy()
            tg_state_dict[tg_key] = Tensor(np_array, requires_grad=False)

    if model.config.quantize in ["nf4", "int8"]:
        assert hasattr(model, 'linear_class') and hasattr(model.linear_class, 'quantize'), \
            "Model must be initialized with a quantizable linear class."
        
        print("[WARNING] Quantizing Qwen3 is experimental. The quantization rules in extra/quantization.py may need adjustment to target MLP and Attention layers correctly.")
        
        device = getattr(model.model.embed_tokens.weight, 'device', Device.DEFAULT)
        tg_state_dict = model.linear_class.quantize(tg_state_dict, device=device)

    for k in tg_state_dict:
        if tg_state_dict[k].dtype != dtypes.uint8:
            tg_state_dict[k] = tg_state_dict[k].cast(dtype)

    print("Assigning weights to model...")
    load_state_dict(model, tg_state_dict, strict=False)

    if model.config.tie_word_embeddings:
        print("Re-tying word embeddings for lm_head...")
        model.lm_head.weight = model.model.embed_tokens.weight
    print("All weights loaded and assigned.")