# src/extra/lora.py

from typing import List

from tinygrad import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters

from model.base_modeling import BaseForCausalLM, BaseAttention
from model.lfm2_modeling import LFM2ForCausalLM # for type hint
from model.qwen3_modeling import Qwen3ForCausalLM # for type hint

class LoRALinear:
    def __init__(self, linear_layer: Linear, r: int, lora_alpha: int):
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        out_features, in_features = self.linear.weight.shape
        self.lora_A = Tensor.kaiming_uniform(in_features, r, requires_grad=True)
        self.lora_B = Tensor.zeros(r, out_features, requires_grad=True)
        self.scaling = lora_alpha / r
    def __call__(self, x: Tensor) -> Tensor:
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def apply_lora_to_model(model: BaseForCausalLM, r: int, alpha: int, target_modules: List[str]):
    print(f"Applying LoRA with r={r}, alpha={alpha} to modules: {target_modules}")
    lora_param_count = 0
    total_param_count = sum(p.numel() for p in get_parameters(model))

    for layer in model.model.layers:
        # Standard path for models like Qwen3
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, BaseAttention):
            attention_module = layer.self_attn
        # LFM2 specific path
        elif hasattr(layer, "operator") and isinstance(layer.operator, BaseAttention):
            attention_module = layer.operator
        else:
            attention_module = None

        if attention_module:
            for module_name in target_modules:
                if hasattr(attention_module, module_name):
                    original_linear = getattr(attention_module, module_name)
                    if isinstance(original_linear, Linear):
                        lora_linear = LoRALinear(original_linear, r, alpha)
                        setattr(attention_module, module_name, lora_linear)
                        lora_param_count += lora_linear.lora_A.numel() + lora_linear.lora_B.numel()

    print(f"Total model parameters: {total_param_count / 1e6:.2f}M")
    print(f"Added {lora_param_count / 1e6:.2f}M trainable LoRA parameters.")

def get_lora_parameters(model: BaseForCausalLM) -> List[Tensor]:
    return [p for p in get_parameters(model) if p.requires_grad]