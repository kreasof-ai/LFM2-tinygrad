# FILE: src/extra/quantization.py
from typing import Tuple
from tinygrad import Tensor, dtypes, Device

def Int8Linear():
  """
  An 8-bit quantized linear layer for tinygrad.
  Weights are quantized to int8, with a per-channel float16 scale.
  """
  class _Int8Linear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported in Int8Linear"
      self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.int8)
      self.scale = Tensor.empty(out_features, dtype=dtypes.float16)

    def __call__(self, x: Tensor) -> Tensor:
      # Dequantize on the fly and perform the dot product.
      # self.weight is (out, in), self.scale is (out,)
      # self.weight.T is (in, out). Scale broadcasts to columns of .T
      return x.dot(self.weight.cast(self.scale.dtype).T * self.scale)

    def dequantize(self) -> Tensor:
      """ Returns the dequantized weight tensor in its original float format. """
      # self.weight is (out, in), self.scale is (out,)
      # To get the original (out, in) weight matrix, we need to transpose the
      # (in, out) dequantized matrix used in the dot product.
      dequantized_for_dot = self.weight.cast(self.scale.dtype).T * self.scale
      return dequantized_for_dot.T

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device, scale_dtype=dtypes.float16) -> dict[str, Tensor]:
        """ Quantizes a state dictionary of FP32/FP16 weights into INT8 format. """
        print("--- Starting INT8 Quantization ---")
        new_state_dict = {}
        # Generalized condition to target linear layers in attention and MLP blocks
        quantizable_substrings = [".self_attn.", ".mlp.", ".operator.", ".feed_forward."]
        for k, v in state_dict.items():
            if any(sub in k for sub in quantizable_substrings) and k.endswith(".weight") and "conv.weight" not in k and "norm.weight" not in k:
                print(f"  Quantizing {k}...")
                v_fp = v.cast(dtypes.float32)
                scale = v_fp.abs().max(axis=1, keepdim=True) / 127.0 + 1e-8
                int8_weight = (v_fp / scale).round().cast(dtypes.int8)
                new_state_dict[k] = int8_weight
                new_state_dict[k.replace(".weight", ".scale")] = scale.squeeze().cast(scale_dtype)
            else:
                new_state_dict[k] = v
        print("--- INT8 Quantization Complete ---")
        return new_state_dict
  return _Int8Linear

def NF4Linear(block_size=64):
  """
  A NormalFloat4 (NF4) quantized linear layer for tinygrad.
  NF4 is a 4-bit quantization scheme with a theoretical information-lossless
  property for normally distributed weights.
  """
  # NF4 quantization values, pre-calculated for efficiency.
  _CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
  ]
  CODE = Tensor(_CODE, dtype=dtypes.float16)

  class _NF4Linear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported in NF4Linear"
      self.in_features, self.out_features = in_features, out_features
      # Weights are packed: two 4-bit values into one uint8.
      self.weight = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      # One scale value per block.
      self.scale = Tensor.empty(int(out_features * in_features / block_size), 1, dtype=dtypes.float16)

    def dequantize(self) -> Tensor:
      """ Returns the dequantized weight tensor in its original float format. """
      # 1. Unpack 4-bit weights from the uint8 tensor.
      high_nibble = self.weight.cast(dtypes.uint8) >> 4
      low_nibble = self.weight.cast(dtypes.uint8) & 0x0F
      
      # 2. Interleave to restore the original order of 4-bit indices.
      unpacked_indices = Tensor.stack(high_nibble, low_nibble, dim=-1).flatten()
      
      # 3. Dequantize: Look up values and apply scaling.
      dequantized_weight = (CODE[unpacked_indices].reshape(-1, block_size) * self.scale).reshape(self.out_features, self.in_features)
      return dequantized_weight

    def __call__(self, x: Tensor) -> Tensor:
      # Dequantize on-the-fly for the linear operation.
      dequantized_weight = self.dequantize()
      return x.linear(dequantized_weight.T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device, scale_dtype=dtypes.float16) -> dict[str, Tensor]:
        """ Quantizes a state dictionary of FP32/FP16 weights into NF4 format. """
        print("--- Starting NF4 Quantization ---")
        new_state_dict = {}
        # Generalized condition to target linear layers in attention and MLP blocks
        quantizable_substrings = [".self_attn.", ".mlp.", ".operator.", ".feed_forward."]
        for k, v in state_dict.items():
            if any(sub in k for sub in quantizable_substrings) and k.endswith(".weight") and "conv.weight" not in k and "norm.weight" not in k:
                print(f"  Quantizing {k}...")
                grouped = v.reshape(-1, block_size)
                scale = grouped.abs().max(axis=1, keepdim=True) + 1e-8
                coded = ((grouped / scale).unsqueeze(-1) - CODE.to(v.device)).abs().argmin(axis=-1).cast(dtypes.uint8).flatten()
                new_state_dict[k] = (coded[::2] << 4) | coded[1::2]
                new_state_dict[k.replace(".weight", ".scale")] = scale.cast(scale_dtype)
            else:
                new_state_dict[k] = v
        print("--- NF4 Quantization Complete ---")
        return new_state_dict
  return _NF4Linear

def SINQLinear(block_size=64, niter=20):
  """
  A Sinkhorn-Normalized Quantization (SINQ) linear layer for tinygrad.
  This method uses dual-scaling (per-row and per-column scales) to better
  handle outliers in weight matrices, improving low-bit quantization performance.
  """
  
  def _sinkhorn_normalize(W: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Applies the Sinkhorn-style normalization loop to a weight matrix tile.
    Returns the normalized matrix and the row/column scales.
    """
    W_norm = W.float().clone() # Use float32 for normalization stability
    
    # Initialize scales. The paper's notation is s for rows, t for columns.
    s_row = Tensor.ones(W.shape[0], dtype=W_norm.dtype)
    s_col = Tensor.ones(W.shape[1], dtype=W_norm.dtype)

    for _ in range(niter):
        # Normalize columns (dim=0)
        col_std = W_norm.std(axis=0) + 1e-6
        W_norm = W_norm / col_std
        s_col = s_col * col_std

        # Normalize rows (dim=1)
        row_std = W_norm.std(axis=1, keepdim=True) + 1e-6
        W_norm = W_norm / row_std
        s_row = s_row * row_std.squeeze()
        
    return W_norm, s_row, s_col

  def _quantize_uniform_affine(W: Tensor, bits: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    A standard round-to-nearest (RTN) uniform quantizer.
    Returns the quantized tensor, scale, and zero-point.
    """
    q_min, q_max = 0, 2**bits - 1
    w_min, w_max = W.min().item(), W.max().item()
    
    scale = (w_max - w_min) / (q_max - q_min) if (w_max - w_min) != 0 else 1.0
    zero_point = q_min - round(w_min / scale)
    zero_point = max(q_min, min(q_max, zero_point)) # clamp

    # Quantize and clamp
    Q = (W / scale).round() + zero_point
    return Q.clamp(q_min, q_max).cast(dtypes.uint8), Tensor([scale]), Tensor([zero_point])

  class _SINQLinear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported in SINQLinear"
      self.in_features, self.out_features = in_features, out_features
      
      # Quantized weights (packed 4-bit)
      self.q = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      
      # SINQ dual scales (one per row/col per block)
      num_blocks = (out_features * in_features) // (block_size * block_size)
      # Assuming square blocks for simplicity, matching paper's recommendation
      rows_per_block, cols_per_block = block_size, block_size
      
      self.s_row = Tensor.empty(num_blocks, rows_per_block, dtype=dtypes.float16)
      self.s_col = Tensor.empty(num_blocks, cols_per_block, dtype=dtypes.float16)

      # Inner scale and zero-point for the uniform quantization (one per block)
      self.inner_scale = Tensor.empty(num_blocks, 1, dtype=dtypes.float16)
      self.inner_zero_point = Tensor.empty(num_blocks, 1, dtype=dtypes.uint8)

    def dequantize(self) -> Tensor:
      """ Dequantizes the entire weight tensor. """
      # 1. Unpack 4-bit weights
      high = (self.q >> 4).flatten()
      low = (self.q & 0x0F).flatten()
      unpacked_q = Tensor.stack(high, low, dim=-1).flatten().reshape(-1, block_size, block_size)
      
      # 2. Dequantize the inner uniform quantized matrix
      W_dequant_inner = (unpacked_q.cast(self.inner_scale.dtype) - self.inner_zero_point) * self.inner_scale
      
      # 3. Apply the SINQ scales
      # s_row shape: (num_blocks, block_size) -> (num_blocks, block_size, 1)
      # s_col shape: (num_blocks, block_size) -> (num_blocks, 1, block_size)
      W_dequant_full = self.s_row.unsqueeze(2) * W_dequant_inner * self.s_col.unsqueeze(1)
      
      # 4. Reshape back to the original matrix shape
      # Assuming out_features is a multiple of block_size
      num_blocks_per_row = self.in_features // block_size
      W_dequant_full = W_dequant_full.reshape(-1, self.in_features)
      
      return W_dequant_full[:self.out_features, :] # Trim any padding

    def __call__(self, x: Tensor) -> Tensor:
      dequantized_weight = self.dequantize()
      return x.linear(dequantized_weight.T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device, scale_dtype=dtypes.float16) -> dict[str, Tensor]:
        print("--- Starting SINQ 4-bit Quantization ---")
        new_state_dict = {}
        quantizable_substrings = [".self_attn.", ".mlp.", ".operator.", ".feed_forward."]
        
        for k, v in state_dict.items():
            if any(sub in k for sub in quantizable_substrings) and k.endswith(".weight") and "conv.weight" not in k and "norm.weight" not in k:
                print(f"  Quantizing {k}...")
                
                # Reshape into blocks (tiles)
                # For simplicity, we assume dimensions are divisible by block_size.
                # Production code might need padding.
                out_features, in_features = v.shape
                assert out_features % block_size == 0 and in_features % block_size == 0, \
                    f"Weight dims ({out_features}, {in_features}) must be divisible by block_size {block_size}"

                tiles = v.reshape(out_features // block_size, block_size, in_features // block_size, block_size).permute(0, 2, 1, 3).reshape(-1, block_size, block_size)
                
                # Process each tile
                q_list, s_row_list, s_col_list, inner_scale_list, inner_z_list = [], [], [], [], []
                for tile in tiles.chunk(tiles.shape[0], dim=0):
                    W_norm, s_row, s_col = _sinkhorn_normalize(tile.squeeze(0))
                    q_tile, inner_scale, inner_z = _quantize_uniform_affine(W_norm, bits=4)
                    
                    q_list.append(q_tile)
                    s_row_list.append(s_row)
                    s_col_list.append(s_col)
                    inner_scale_list.append(inner_scale)
                    inner_z_list.append(inner_z)

                # Pack 4-bit Q values into uint8
                full_q = Tensor.stack(*q_list).flatten()
                packed_q = (full_q[::2] << 4) | full_q[1::2]

                # Store all the new tensors in the state dict
                new_state_dict[k.replace(".weight", ".q")] = packed_q
                new_state_dict[k.replace(".weight", ".s_row")] = Tensor.stack(*s_row_list).cast(scale_dtype)
                new_state_dict[k.replace(".weight", ".s_col")] = Tensor.stack(*s_col_list).cast(scale_dtype)
                new_state_dict[k.replace(".weight", ".inner_scale")] = Tensor.stack(*inner_scale_list).cast(scale_dtype)
                new_state_dict[k.replace(".weight", ".inner_zero_point")] = Tensor.stack(*inner_z_list).cast(dtypes.uint8)
            else:
                new_state_dict[k] = v
        print("--- SINQ Quantization Complete ---")
        return new_state_dict
  return _SINQLinear