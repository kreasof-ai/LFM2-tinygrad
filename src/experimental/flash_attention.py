from typing import List, Optional

from tinygrad import Tensor, dtypes

def flash_attn(q: Tensor, k: Tensor, v: Tensor, sm_scale: float,
               causal: bool = True, seq_lens: Optional[List[int]] = None,
               start_pos: int = 0) -> Tensor:
    """
    Flash attention for tinygrad with
      - padding to multiple of tile size
      - causal mask
      - support for GQA (k/v have different n_heads via repeat_kv)
    q: (B, Hq,  S, D)
    k: (B, Hkv, S, D)  ← already repeated to match Hq if GQA
    v: (B, Hkv, S, D)
    seq_lens: actual lengths per sample (for causal trimming on padded batch)
    Returns (B, Hq, S, D)
    """
    B, H, S, D = q.shape
    Br, Bc = 64, 64
    pad = (Br - S % Br) % Br
    if pad:
        q = q.pad((0, 0, 0, pad, 0, 0, 0, 0))   # (B,H,S+pad,D)
        k = k.pad((0, 0, 0, pad, 0, 0, 0, 0))
        v = v.pad((0, 0, 0, pad, 0, 0, 0, 0))
    S_padded = q.shape[2]

    # collect tiles instead of in-place assignment
    O_tiles = []
    for i in range(0, S_padded, Br):
        qi = q[:, :, i:i+Br, :]                 # (B,H,Br,D)
        mi = Tensor.full((B, H, Br), -float('inf'))
        li = Tensor.zeros((B, H, Br))
        Oi = Tensor.zeros((B, H, Br, D))
        for j in range(0, S_padded, Bc):
            kj = k[:, :, j:j+Bc, :]
            vj = v[:, :, j:j+Bc, :]
            Sij = (qi @ kj.transpose(-2, -1)) * sm_scale          # (B,H,Br,Bc)

            # ---- causal mask ----
            if causal:
                # absolute positions in the full sequence
                query_pos = start_pos + i + Tensor.arange(Br).reshape(1, 1, Br, 1)
                key_pos   = j + Tensor.arange(Bc).reshape(1, 1, 1, Bc)
                causal_mask = query_pos < key_pos          # (1,1,Br,Bc)
                Sij = Sij.masked_fill(causal_mask, -float('inf'))

            # ---- per-sample length mask (for ragged batches) ----
            if seq_lens is not None:
                lens_t = Tensor(seq_lens, dtype=dtypes.int32).reshape(B, 1, 1, 1)
                col_ids_b = Tensor.arange(j, j+Bc).reshape(1, 1, 1, Bc)
                len_mask = col_ids_b >= lens_t
                Sij = Sij.masked_fill(len_mask, -float('inf'))

            # online softmax
            mij = Sij.max(axis=-1, keepdim=True)
            Pij = (Sij - mij).exp()
            lij = Pij.sum(axis=-1, keepdim=True)
            mi_new = Tensor.maximum(mi, mij.squeeze(-1))
            alpha = (mi - mi_new).exp()
            beta  = (mij.squeeze(-1) - mi_new).exp()
            Oi = Oi * alpha.unsqueeze(-1) + (Pij @ vj) * beta.unsqueeze(-1)
            li = li * alpha + lij.squeeze(-1) * beta
            mi = mi_new
        O_tiles.append(Oi / li.unsqueeze(-1))

    # concatenate along sequence dimension
    O = Tensor.cat(*O_tiles, dim=2)          # (B, H, S_padded, D)
    return O[:, :, :S, :]                    # remove padding
