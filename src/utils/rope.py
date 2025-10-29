# src/utils/rope.py

import math
from typing import Tuple, Optional
from tinygrad import Tensor

def _precompute_rope_cache(dim: int, max_seq_len: int, base: float, dtype) -> Tuple[Tensor, Tensor]:
    """Pre-computes the rotary positional embeddings for the given dimensions."""
    inv_freq = 1.0 / (base ** (Tensor.arange(0, dim, 2, dtype=dtype) / dim))
    t = Tensor.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = t.reshape(-1, 1) * inv_freq.reshape(1, -1)
    emb = Tensor.cat(freqs, freqs, dim=-1)
    return emb.cos().contiguous(), emb.sin().contiguous()

def _precompute_rope_cache_llama3(dim: int, max_seq_len: int, rope_scaling: dict, dtype) -> Tuple[Tensor, Tensor]:
    """Pre-computes the rotary positional embeddings using Llama 3's specific scaling."""
    base = rope_scaling["rope_theta"]
    
    # Standard RoPE inverse frequencies
    inv_freq = 1.0 / (base ** (Tensor.arange(0, dim, 2, dtype=dtype) / dim))

    # Llama 3 scaling parameters
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]
    
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    # Calculate wavelength for each frequency
    wavelen = 2 * math.pi / inv_freq

    # Handle low frequencies (long wavelengths) - scale them down
    inv_freq_llama = (wavelen > low_freq_wavelen).where(inv_freq / factor, inv_freq)

    # Handle medium frequencies - interpolate between scaled and original
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    
    is_medium_freq = ((wavelen < high_freq_wavelen).logical_not()) * ((wavelen > low_freq_wavelen).logical_not())
    inv_freq_llama = is_medium_freq.where(smoothed_inv_freq, inv_freq_llama)

    # Compute final embeddings
    t = Tensor.arange(max_seq_len, dtype=inv_freq_llama.dtype)
    freqs = t.reshape(-1, 1) * inv_freq_llama.reshape(1, -1)
    emb = Tensor.cat(freqs, freqs, dim=-1)
    return emb.cos().contiguous(), emb.sin().contiguous()


def rotate_half(x: Tensor): return Tensor.cat(-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2], dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin): return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)