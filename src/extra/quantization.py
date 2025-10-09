# FILE: src/extra/quantization.py
from tinygrad import Tensor, dtypes, Device

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

    def __call__(self, x: Tensor) -> Tensor:
      # 1. Unpack 4-bit weights from the uint8 tensor.
      high_nibble = self.weight.cast(dtypes.uint8) >> 4
      low_nibble = self.weight.cast(dtypes.uint8) & 0x0F
      
      # 2. Interleave to restore the original order of 4-bit indices.
      unpacked_indices = Tensor.stack(high_nibble, low_nibble, dim=-1).flatten()
      
      # 3. Dequantize: Look up values and apply scaling.
      dequantized_weight = (CODE[unpacked_indices].reshape(-1, block_size) * self.scale).reshape(self.out_features, self.in_features)
      
      # 4. Perform the linear operation.
      return x.linear(dequantized_weight.T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device, scale_dtype=dtypes.float16) -> dict[str, Tensor]:
      """
      Quantizes a state dictionary of FP32/FP16 weights into NF4 format.
      """
      print("--- Starting NF4 Quantization ---")
      new_state_dict = {}
      for k, v in state_dict.items():
        # Condition to select which layers to quantize.
        # For LFM2, linear layers are in 'operator' (attention/conv) or 'feed_forward' blocks.
        
        # --- FIX START ---
        # The original condition was too broad. This new condition explicitly excludes 'conv.weight'.
        if (".operator." in k or ".feed_forward." in k) and k.endswith(".weight") and "conv.weight" not in k and "norm.weight" not in k:
        # --- FIX END ---
            print(f"  Quantizing {k}...")
            # Reshape into blocks for quantization.
            grouped = v.reshape(-1, block_size)
            
            # Calculate scale (absmax) for each block. Add epsilon for stability.
            scale = grouped.abs().max(axis=1, keepdim=True) + 1e-8
            
            # Normalize, find the closest NF4 value, and cast to uint8.
            coded = ((grouped / scale).unsqueeze(-1) - CODE.to(v.device)).abs().argmin(axis=-1).cast(dtypes.uint8).flatten()
            
            # Pack two 4-bit values into one uint8 byte.
            new_state_dict[k] = (coded[::2] << 4) | coded[1::2]
            new_state_dict[k.replace(".weight", ".scale")] = scale.cast(scale_dtype)
        else:
            new_state_dict[k] = v # Pass through non-quantized tensors (embeddings, norms, etc.).
      print("--- NF4 Quantization Complete ---")
      return new_state_dict
  return _NF4Linear