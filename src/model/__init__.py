# src/model/__init__.py

__all__ = ["base_modeling", "lfm2_modeling", "qwen3_modeling", "qwen2_modeling", "gemma3_modeling"]

# Import the submodules
from . import base_modeling
from . import lfm2_modeling
from . import qwen3_modeling
from . import qwen2_modeling
from . import gemma3_modeling

MODEL_MAP = {
    "LFM2": lfm2_modeling.LFM2ForCausalLM,
    "Qwen3": qwen3_modeling.Qwen3ForCausalLM,
    "Qwen2": qwen2_modeling.Qwen2ForCausalLM,
    "Qwen2.5": qwen2_modeling.Qwen2ForCausalLM,
    "Gemma3": gemma3_modeling.Gemma3ForCausalLM,
}