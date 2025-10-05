# LFM2 (Liquid Foundation Model 2) tinygrad Implementation

This version is adapted to load pretrained weights from Hugging Face Hub.

## Get started

- Install dependencies:

    `pip install tinygrad torch transformers huggingface_hub safetensors tqdm`

- Starts `main.py`:

    `python main.py`

- You'll see this output

```
--- Loading LFM2 Model: LiquidAI/LFM2-350M ---

Model configuration:
Layer 0: Convolution
Layer 1: Convolution
Layer 2: Attention
Layer 3: Convolution
Layer 4: Convolution
Layer 5: Attention
Layer 6: Convolution
Layer 7: Convolution
Layer 8: Attention
Layer 9: Convolution
Layer 10: Attention
Layer 11: Convolution
Layer 12: Attention
Layer 13: Convolution
Layer 14: Attention
Layer 15: Convolution

Initializing model architecture...
Fetching weights from LiquidAI/LFM2-350M/model.safetensors...
Loading and assigning weights...
Re-tying word embeddings for lm_head...
All weights loaded and assigned.

Loading tokenizer...

--- Starting Text Generation ---
Formatted Prompt (decoded): <|startoftext|><|im_start|>user
The secret to a long and happy life is<|im_end|>
<|im_start|>assistant

Processing prompt...
Generating new tokens...
<|startoftext|><|im_start|>user
The secret to a long and happy life is<|im_end|>
<|im_start|>assistant
 secret to a long and happy life often involves a combination of several key elements, including:

1. **Health and Well-being**: Regular exercise, a balanced diet, adequate sleep, and stress management are crucial. Physical health directly impacts mental
```


## Acknowledgment

> Heavily inspired from https://github.com/kyegomez/LFM2 and official https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py implementation

## Disclaimer

- Empirical test with `debug_prefilling.py` shows huggingface implementation apply final norm inside final layer.

This is output when final norm is applied inside layer 15

```
--- Starting Step-by-Step tinygrad Comparison ---
--- Comparing: Initial Embeddings ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000040, PT=0.000040
  Max absolute difference: 0.000000
  ✅ MATCH: Tensors are numerically close.
----------------------------------------
--- Comparing: RoPE Cos ---
  Shapes: TG=(1, 4, 64), PT=(1, 4, 64)
  Means:  TG=0.936163, PT=0.936163
  Max absolute difference: 0.000000
  ✅ MATCH: Tensors are numerically close.
------------------------------
--- Comparing: RoPE Sin ---
  Shapes: TG=(1, 4, 64), PT=(1, 4, 64)
  Means:  TG=0.086091, PT=0.086091
  Max absolute difference: 0.000000
  ✅ MATCH: Tensors are numerically close.
------------------------------
--- Comparing: Layer 0 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.000293, PT=-0.000293
  Max absolute difference: 0.000000
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 1 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.000917, PT=-0.000917
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 2 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000082, PT=0.000082
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 3 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000129, PT=0.000129
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 4 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.000286, PT=-0.000286
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 5 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000247, PT=0.000247
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 6 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000444, PT=0.000444
  Max absolute difference: 0.000002
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 7 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006097, PT=-0.006097
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 8 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006024, PT=-0.006024
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 9 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006152, PT=-0.006152
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 10 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006291, PT=-0.006291
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 11 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006382, PT=-0.006382
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 12 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006064, PT=-0.006064
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 13 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006869, PT=-0.006869
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 14 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.003247, PT=-0.003247
  Max absolute difference: 0.000070
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 15 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.005748, PT=-0.005748
  Max absolute difference: 0.000732
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Final Logits ---
  Shapes: TG=(1, 4, 65536), PT=(1, 4, 65536)
  Means:  TG=-2.157017, PT=-2.157017
  Max absolute difference: 0.000174
  ✅ MATCH: Tensors are numerically close.
----------------------------------

🎉🎉🎉 All checks passed! The models match perfectly. 🎉🎉🎉
```

And this when it's applied after layer 15:
```
--- Starting Step-by-Step tinygrad Comparison ---
--- Comparing: Initial Embeddings ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000040, PT=0.000040
  Max absolute difference: 0.000000
  ✅ MATCH: Tensors are numerically close.
----------------------------------------
--- Comparing: RoPE Cos ---
  Shapes: TG=(1, 4, 64), PT=(1, 4, 64)
  Means:  TG=0.936163, PT=0.936163
  Max absolute difference: 0.000000
  ✅ MATCH: Tensors are numerically close.
------------------------------
--- Comparing: RoPE Sin ---
  Shapes: TG=(1, 4, 64), PT=(1, 4, 64)
  Means:  TG=0.086091, PT=0.086091
  Max absolute difference: 0.000000
  ✅ MATCH: Tensors are numerically close.
------------------------------
--- Comparing: Layer 0 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.000293, PT=-0.000293
  Max absolute difference: 0.000000
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 1 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.000917, PT=-0.000917
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 2 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000082, PT=0.000082
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 3 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000129, PT=0.000129
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 4 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.000286, PT=-0.000286
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 5 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000247, PT=0.000247
  Max absolute difference: 0.000001
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 6 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=0.000444, PT=0.000444
  Max absolute difference: 0.000002
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 7 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006097, PT=-0.006097
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 8 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006024, PT=-0.006024
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 9 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006152, PT=-0.006152
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
------------------------------------
--- Comparing: Layer 10 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006291, PT=-0.006291
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 11 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006382, PT=-0.006382
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 12 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006064, PT=-0.006064
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 13 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.006869, PT=-0.006869
  Max absolute difference: 0.000067
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 14 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.003247, PT=-0.003247
  Max absolute difference: 0.000070
  ✅ MATCH: Tensors are numerically close.
-------------------------------------
--- Comparing: Layer 15 Output ---
  Shapes: TG=(1, 4, 1024), PT=(1, 4, 1024)
  Means:  TG=-0.000973, PT=-0.005748
  Max absolute difference: 29.115623
  ❌ MISMATCH: Tensors are NOT close.
-------------------------------------

‼️ DIVERGENCE DETECTED AT LAYER 15 ‼️
```