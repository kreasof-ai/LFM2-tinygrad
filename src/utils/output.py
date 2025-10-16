from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from tinygrad import Tensor

@dataclass
class CausalLMOutputWithPast:
    """
    Base class for causal language model (or autoregressive) outputs.
    Adapted for tinygrad from Hugging Face's CausalLMOutputWithPast.
    """
    logits: Tensor
    loss: Optional[Tensor] = None
    past_key_values: Optional[List[Any]] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
