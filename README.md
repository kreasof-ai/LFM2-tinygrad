# LFM2 (Liquid Foundation Model 2) with Tinygrad

- **What is Liquid Foundation Model 2?**

   It's a sequence model architecture from [Liquid AI](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models) synthesized from [STAR Framework](https://www.liquid.ai/research/automated-architecture-synthesis-via-targeted-evolution). It's combination of specialized short conv layer and [Grouped Query Attention](https://arxiv.org/abs/2305.13245).

- **What is Tinygrad?**

    Tinygrad is a minimalist tensor and deep learning library developed by [George Hotz](https://github.com/geohot) and heavily maintained by [tiny corp](https://tinygrad.org/). Tinygrad offers extreme design simplicity and readibility compared to PyTorch. As stated in their official codebase:

    > Due to its extreme simplicity, it is the easiest framework to add new accelerators to, with support for both inference and training. If XLA is CISC, tinygrad is RISC.

- **Does this project actually can run?**

    This shit works. It's just very slow and you can't just calling TinyJit because JIT compilation in tinygrad only support fixed size input. We still thinking workaround to speedup the inference. 
    
    Unironically, training is much faster than you think (and even faster than inference) because training loop only require fixed size shape kernel. Training speed can vary between cards, but in our testing with RX 6700 XT (max_length=512,bsz=2,max_steps=100), it can be done in 4m 20s (and even faster after first compilation) with around 10.1/12GB memory utilization. This suggest Tinygrad is actually reasonably fast after compilation.

     <img width="1920" height="1080" alt="Screenshot 2025-10-06 190415" src="https://github.com/user-attachments/assets/861afba5-ab6e-4dff-ab50-424e5cb0a56d" />


- **What is the goal of this project?**

    Current benefit is mostly educational. Implementing existing architecture in existing software stack (PyTorch) and infrastucture (CUDA) is one thing. But generalize your understanding beyond that is completely different territory. This project is a proof of concept that you can transfer cutting edge concept and give you feeling of control that you finally understand to build unique concept from scratch.

## Get started

- Install dependencies:

    `pip install tinygrad torch transformers huggingface_hub safetensors tqdm datasets`

- Starts `run.py`:

    `python src/run.py`

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
Processing prompt...
Generating new tokens...
<|startoftext|><|im_start|>user
The secret to a long and happy life is<|im_end|>
<|im_start|>assistant
The secret to a long and happy life is often attributed to a combination of several key elements, including:

1. **Health and Well-being**: Maintaining good physical health through regular exercise, a balanced diet, and adequate sleep is crucial.

--- Generation Complete ---
```

## Paged Attention

Current Paged Attention implementation kinda works, but not doing anything to speedup the inference. It's in fact slightly slower than naive.

```
Starting LFM2 Inference Speed Test
Prompt: 'The secret to a long and happy life is'
Tokens to generate: 128
tinygrad Device: GPU
--------------------------------------------------

--- 1. Testing Hugging Face (PyTorch) Reference ---
Loading model... (This might take a moment)
`torch_dtype` is deprecated! Use `dtype` instead!
Generating tokens...
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Generated Text Sample: The secret to a long and happy life is a combination of various factors, includi...
Time taken: 4.3388 seconds
Tokens per second: 29.50 tok/s

--- 2. Testing Naive tinygrad Implementation ---
Loading model...
Fetching weights from LiquidAI/LFM2-350M/model.safetensors...
Loading and assigning weights...
Re-tying word embeddings for lm_head...
All weights loaded and assigned.
Generating tokens...

--- Starting Text Generation ---
Processing prompt...
Generating new tokens...
<|startoftext|><|im_start|>user
The secret to a long and happy life is<|im_end|>
<|im_start|>assistant
The secret to a long and happy life is a combination of various factors, including:

1. **Health and Well-being**: Regular exercise, a balanced diet, and adequate sleep are fundamental.
2. **Mindfulness and Mental Health**: Practicing mindfulness, meditation, and therapy can help manage stress and maintain emotional balance.
3. **Relationships**: Strong, supportive relationships with family and friends provide emotional support and a sense of belonging.
4. **Personal Growth**: Continuous learning and personal development can keep life engaging and fulfilling.
5. **Purpose and Meaning**: Finding purpose in life through hobbies
--- Generation Complete ---

Time taken: 343.5441 seconds
Tokens per second: 0.37 tok/s

--- 3. Testing tinygrad with Paged Attention ---
Loading model...
Fetching weights from LiquidAI/LFM2-350M/model.safetensors...
Loading and assigning weights...
Re-tying word embeddings for lm_head...
All weights loaded and assigned.
Generating tokens...

--- Starting Text Generation ---
Allocated batch slot: 0
Reserved memory for a maximum sequence length of 146
Processing prompt...
Generating new tokens...
<|startoftext|><|im_start|>user
The secret to a long and happy life is<|im_end|>
<|im_start|>assistant
The secret to a long and happy life is a combination of various factors, including:

1. **Health and Well-being**: Regular exercise, a balanced diet, and adequate sleep are fundamental.
2. **Mindfulness and Mental Health**: Practicing mindfulness, meditation, and therapy can help manage stress and maintain emotional balance.
3. **Relationships**: Strong, supportive relationships with family and friends provide emotional support and a sense of belonging.
4. **Personal Growth**: Continuous learning and personal development can keep life engaging and fulfilling.
5. **Purpose and Meaning**: Finding purpose in life through hobbies
--- Generation Complete ---

Time taken: 364.7106 seconds
Tokens per second: 0.35 tok/s


==================================================
           INFERENCE SPEED TEST SUMMARY
==================================================
Implementation            | Time Taken (s)  | Tokens/sec
--------------------------------------------------
Hugging Face (PyTorch)    | 4.3388          | 29.50
Naive tinygrad            | 343.5441        | 0.37
Paged tinygrad            | 364.7106        | 0.35
==================================================
```

## Disclaimer

- Empirical test with `src/debug_prefilling.py` shows huggingface implementation apply final norm inside final layer.

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

## Acknowledgment

> Heavily inspired from https://github.com/kyegomez/LFM2 and official https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py implementation
