from typing import List

# tinygrad imports
from tinygrad import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters

from model.lfm2_modeling import LFM2ForCausalLM, GroupedQueryAttention

class LoRALinear:
    """
    Wraps a tinygrad Linear layer with LoRA functionality.
    The original weights are frozen, and two new low-rank matrices (A and B)
    are made trainable.
    """
    def __init__(self, linear_layer: Linear, r: int, lora_alpha: int):
        self.linear = linear_layer
        self.linear.weight.requires_grad = False

        out_features, in_features = self.linear.weight.shape

        self.lora_A = Tensor.kaiming_uniform(in_features, r, requires_grad=True)
        self.lora_B = Tensor.zeros(r, out_features, requires_grad=True)
        
        self.scaling = lora_alpha / r

    def __call__(self, x: Tensor) -> Tensor:
        original_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out

def apply_lora_to_model(model: LFM2ForCausalLM, r: int, alpha: int, target_modules: List[str]):
    """
    Recursively traverses the model and replaces target Linear layers with LoRALinear layers.
    """
    print(f"Applying LoRA with r={r}, alpha={alpha} to modules: {target_modules}")
    lora_param_count = 0
    total_param_count = sum(p.numel() for p in get_parameters(model))

    for layer in model.model.layers:
        if isinstance(layer.operator, GroupedQueryAttention):
            for module_name in target_modules:
                if hasattr(layer.operator, module_name):
                    original_linear = getattr(layer.operator, module_name)
                    if isinstance(original_linear, Linear):
                        lora_linear = LoRALinear(original_linear, r, alpha)
                        setattr(layer.operator, module_name, lora_linear)
                        lora_param_count += lora_linear.lora_A.numel() + lora_linear.lora_B.numel()

    print(f"Total model parameters: {total_param_count / 1e6:.2f}M")
    print(f"Added {lora_param_count / 1e6:.2f}M trainable LoRA parameters.")

def get_lora_parameters(model: LFM2ForCausalLM) -> List[Tensor]:
    """
    Returns a list of all trainable parameters in the model, which after applying
    LoRA, should only be the lora_A and lora_B matrices.
    """
    return [p for p in get_parameters(model) if p.requires_grad]