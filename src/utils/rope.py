from typing import Tuple
from tinygrad import Tensor

def _precompute_rope_cache(dim: int, max_seq_len: int, base: float, dtype) -> Tuple[Tensor, Tensor]:
    """Pre-computes the rotary positional embeddings for the given dimensions."""
    inv_freq = 1.0 / (base ** (Tensor.arange(0, dim, 2, dtype=dtype) / dim))
    t = Tensor.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = t.reshape(-1, 1) * inv_freq.reshape(1, -1)
    emb = Tensor.cat(freqs, freqs, dim=-1)
    return emb.cos().contiguous(), emb.sin().contiguous()

def rotate_half(x: Tensor): return Tensor.cat(-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2], dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin): return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
