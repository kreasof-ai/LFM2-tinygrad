# LFM2 (Liquid Foundation Model 2) in tinygrad

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/license/apache-2-0)

This repository contains a from-scratch implementation of the **[LFM2 (Liquid Foundation Model 2)](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)** architecture using **[tinygrad](https://github.com/tinygrad/tinygrad)**, the minimalist deep learning framework. The project focuses on educational value, numerical correctness against official implementations, and exploring advanced features like quantization, paged attention, and LoRA for supervised fine-tuning.

---

## 🌟 Core Features

-   ✅ **Numerically Correct Implementation**: Rigorously verified against the official Hugging Face Transformers implementation to ensure matching outputs for both standard and paged attention modes (see *Implementation Notes*).
-   🚀 **Inference Ready**: Includes scripts for running text generation with the pretrained `LiquidAI/LFM2-350M` model.
-   💡 **Quantization Ready**: Built-in support for **INT8** and **NormalFloat4 (NF4)** quantization, allowing for a reduced memory footprint during inference.
-   🔬 **Experimental Paged Attention**: An implementation of paged attention to explore efficient memory management for the KV cache, inspired by vLLM.
-   🎓 **Supervised Fine-Tuning (SFT)**: A complete training script (`src/train_sft.py`) is provided, enabling you to fine-tune LFM2 on your own datasets.
-   🔧 **LoRA Support**: The training script includes built-in support for **Low-Rank Adaptation (LoRA)**, allowing for efficient, low-memory fine-tuning of the model.

---

## ✅ Supported Models

Thanks to a flexible configuration loader, this implementation supports these models:

-   [`LiquidAI/LFM2-350M`](https://huggingface.co/LiquidAI/LFM2-350M) (Default)
-   [`LiquidAI/LFM2-700M`](https://huggingface.co/LiquidAI/LFM2-700M)
-   [`LiquidAI/LFM2-1.2B`](https://huggingface.co/LiquidAI/LFM2-1.2B)
-   [`LiquidAI/LFM2-2.6B`](https://huggingface.co/LiquidAI/LFM2-2.6B)

---

## 🧐 Project Status & Performance

The primary goal of this project is educational: to demonstrate how a cutting-edge architecture can be built and understood in a framework other than PyTorch/TensorFlow.

### Inference Performance
Currently, inference is **very slow**. This is an expected limitation of tinygrad's current Just-In-Time (JIT) compiler, which is optimized for tensors with **fixed shapes**. Autoregressive decoding, where the sequence length changes at each step, requires re-compiling the computation graph for every new token, hindering performance.

Here is a performance comparison (model:`LiquidAI/LFM2-350M`) against the official PyTorch implementation on an AMD RX 6700 XT, including new quantization modes:

| Implementation                 | Time Taken (s) for 64 tokens | Tokens/sec |
| ------------------------------ | ---------------------------- | ---------- |
| Hugging Face (PyTorch)         | 2.6467                       | 24.18      |
| Standard tinygrad (FP32)       | 65.3724                      | 0.98       |
| Standard tinygrad (FP16)       | 71.8813                      | 0.89       |
| Standard tinygrad (INT8)       | 63.3681                      | 1.01       |
| Paged tinygrad (FP32)          | 77.2624                      | 0.83       |
| Paged tinygrad (FP16)          | 102.9025                     | 0.62       |
| Paged tinygrad (INT8)          | 83.5189                      | 0.77       |

*As shown, the bottleneck remains in kernel compilation, not memory access or data type. Even with quantization, the speed benefits are minimal due to overhead.*

### Training Performance
In contrast, **training performance is surprisingly competitive**. Since the training loop uses fixed-size input batches (`batch_size`, `max_length`), the tinygrad JIT can compile highly optimized kernels once and reuse them.

In our testing on an **AMD RX 6700 XT**, a short fine-tuning run (`max_length=512`, `batch_size=2`, `max_steps=100`) completed in approximately **4 minutes and 20 seconds**, utilizing around 10.1/12GB of VRAM. This demonstrates that tinygrad is a capable framework for training when tensor shapes are static.

<img width="1920" height="1080" alt="Screenshot of a successful training run" src="https://github.com/user-attachments/assets/861afba5-ab6e-4dff-ab50-424e5cb0a56d" />

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies.

```bash
git clone https://github.com/kreasof-ai/LFM2-tinygrad.git
cd LFM2-tinygrad
pip install tinygrad torch transformers huggingface_hub safetensors tqdm datasets wandb
```

### 2. Usage

#### Standard Inference
To run standard text generation, use `run.py`. This script loads the pretrained LFM2-350M model and generates text from a sample prompt.

```bash
python src/run.py
```

#### Inference with Quantization
You can enable NF4 or INT8 quantization to reduce memory usage with the `--quantize` flag.

```bash
# Run with 4-bit NormalFloat quantization
python src/run.py --quantize nf4

# Run with 8-bit Integer quantization
python src/run.py --quantize int8
```

#### Supervised Fine-Tuning (SFT) with LoRA
The `train_sft.py` script allows you to fine-tune the model on any conversational dataset from the Hugging Face Hub. LoRA is enabled by default for efficiency. **Note: Quantization is not supported for training.**

Here is an example command to run a short training job on the `mlabonne/FineTome-100k` dataset:

```bash
python src/train_sft.py \
    --model_id "LiquidAI/LFM2-350M" \
    --dataset_id "mlabonne/FineTome-100k" \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --batch_size 2 \
    --max_length 512 \
    --learning_rate 1e-4 \
    --max_steps 100 \
    --use_wandb \
    --wandb_project "lfm2-tinygrad-sft"
```
*   To perform a full fine-tune (instead of LoRA), remove the `--use_lora` flag.
*   The script automatically handles data processing, masking labels for prompts, and logging to Weights & Biases (if `--use_wandb` is specified).

---

## 🛠️ Implementation Notes

### Numerical Verification
This implementation has been carefully verified against the official Hugging Face LFM2 model. The debugging scripts (`src/debug_prefilling.py` and `src/debug_decoding.py`) perform a layer-by-layer comparison of hidden states and cache values for both **standard and paged attention modes**, confirming that the tinygrad model produces numerically identical outputs.

### Final Layer Norm
Our analysis during debugging revealed a subtle but critical implementation detail in the official model: the final `RMSNorm` layer is applied **inside the final decoder block**, right before the output is passed to the language model head. This repository's implementation correctly mirrors that structure. Applying the norm *after* the final block would lead to a numerical mismatch.

The output below from `debug_prefilling.py` shows a perfect match across all layers and final logits, confirming the correctness of our model structure.

```
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

---

## 🗺️ Roadmap

-   [ ] **Improve Inference Speed:** Investigate workarounds for the dynamic shape problem, potentially by padding to fixed sequence length buckets or exploring future tinygrad features.
-   [ ] **Optimize Paged Attention:** Refine the paged attention CUDA/Metal kernels once the core JIT issues are addressed.
-   [ ] **Integrate Flash Attention:** Integrate the existing `flash_attention.py` implementation into the main model as an optional, high-performance attention mechanism.

## Acknowledgments
This project was heavily inspired by the following resources:
-   [kyegomez/LFM2](https://github.com/kyegomez/LFM2) for an early PyTorch implementation.
-   The official [Hugging Face Transformers implementation of LFM2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py).