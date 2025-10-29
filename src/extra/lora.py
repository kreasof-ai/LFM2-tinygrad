# src/extra/lora.py

from typing import List

from tinygrad import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters

class LoRALinear:
    def __init__(self, linear_layer: Linear, r: int, lora_alpha: int):
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if hasattr(self.linear, 'bias') and self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        out_features, in_features = self.linear.weight.shape
        self.lora_A = Tensor.kaiming_uniform(in_features, r, requires_grad=True)
        self.lora_B = Tensor.zeros(r, out_features, requires_grad=True)
        self.scaling = lora_alpha / r
    def __call__(self, x: Tensor) -> Tensor:
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling
    def merge_weights(self) -> Tensor:
        """
        Merges the LoRA weights with the base linear layer's weights.
        Handles dequantization of the base layer if necessary.
        """
        # Get the original weight, dequantizing it if it's from an Int8/NF4 layer.
        original_weight = self.linear.dequantize() if hasattr(self.linear, 'dequantize') else self.linear.weight
        
        # Calculate the LoRA delta and transpose it to match the original weight's shape.
        # Original weight shape: (out_features, in_features)
        # lora_A @ lora_B shape: (in_features, out_features)
        lora_delta = (self.lora_A @ self.lora_B).T * self.scaling
        
        # Add the delta to the original weight.
        return (original_weight + lora_delta).realize()

def apply_lora_to_model(model, r: int, alpha: int, target_modules: List[str]):
    print(f"Applying LoRA with r={r}, alpha={alpha} to modules: {target_modules}")
    lora_param_count = 0
    total_param_count = sum(p.numel() for p in get_parameters(model))

    for layer in model.model.layers:
        # Standard path for models like Qwen3
        if hasattr(layer, "self_attn"):
            attention_module = layer.self_attn
        # LFM2 specific path
        elif hasattr(layer, "operator"):
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

def get_lora_parameters(model) -> List[Tensor]:
    return [p for p in get_parameters(model) if p.requires_grad]