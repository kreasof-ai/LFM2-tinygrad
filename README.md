# LFM2 and Qwen3 in tinygrad

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/license/apache-2-0)

This repository contains from-scratch implementations of the **[LFM2 (Liquid Foundation Model 2)](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)** and **[Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)** architectures using **[tinygrad](https://github.com/tinygrad/tinygrad)**.

Built upon a **unified, extensible base modeling architecture**, this project serves as an educational tool to demonstrate how modern LLMs can be built, fine-tuned, and verified for numerical correctness. It explores advanced features like quantization, paged attention, and LoRA, with a forward-looking goal of supporting diverse model types on commodity and non-NVIDIA GPUs.

---

## 🌟 Core Features

-   ✅ **Unified Architecture**: Built on a modular `base_modeling.py` that simplifies the adaptation of new Hugging Face models.
-   ✅ **Numerically Correct Implementations**: Rigorously verified against official Hugging Face Transformers implementations to ensure matching outputs.
-   🚀 **Multi-Model Support**: Ready-to-use implementations for both LFM2 and Qwen3 model families.
-   🎓 **Supervised Fine-Tuning (SFT)**: A complete training script (`src/train_sft.py`) is provided, enabling you to fine-tune models on your own datasets.
-   🔧 **LoRA Support**: Built-in support for **Low-Rank Adaptation (LoRA)**, allowing for efficient, low-memory fine-tuning.
-   💡 **Quantization Ready**: Support for **INT8** and **NormalFloat4 (NF4)** quantization to reduce memory footprint during inference.
-   🔬 **Advanced Attention Mechanisms**: Includes experimental implementations of **Paged Attention** (for efficient KV cache management) and **Flash Attention**.

---

## ✅ Supported Models

Thanks to a flexible configuration loader, this implementation supports these model families:

### LFM2

-   [`LiquidAI/LFM2-350M`](https://huggingface.co/LiquidAI/LFM2-350M) (Default)
-   [`LiquidAI/LFM2-700M`](https://huggingface.co/LiquidAI/LFM2-700M)
-   [`LiquidAI/LFM2-1.2B`](https://huggingface.co/LiquidAI/LFM2-1.2B)
-   [`LiquidAI/LFM2-2.6B`](https://huggingface.co/LiquidAI/LFM2-2.6B)

### Qwen2

This implementation should be compatible with all dense Qwen2 models. We mark version with explicit testing.

-   [`Qwen/Qwen2-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) (Tested ✅)
-   [`Qwen/Qwen2-0.5B`](https://huggingface.co/Qwen/Qwen2-0.5B) (Tested ✅)
-   [`Qwen/Qwen2-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) (Tested ✅)
-   [`Qwen/Qwen2-1.5B`](https://huggingface.co/Qwen/Qwen2-1.5B) (Untested)
-   [`Qwen/Qwen2-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-7B-Instruct) (Untested)
-   [`Qwen/Qwen2-7B`](https://huggingface.co/Qwen/Qwen2-7B) (Untested)
-   [`Qwen/Qwen2-72B-Instruct`](https://huggingface.co/Qwen/Qwen2-7B-Instruct) (Untested)
-   [`Qwen/Qwen2-72B`](https://huggingface.co/Qwen/Qwen2-7B) (Untested)

### Qwen2.5

This implementation should be compatible with all dense Qwen2.5 models. We mark version with explicit testing.

-   [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Tested ✅)
-   [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) (Untested)
-   [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Untested)
-   [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) (Untested)
-   [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Untested)
-   [`Qwen/Qwen2.5-3B`](https://huggingface.co/Qwen/Qwen2.5-3B) (Untested)
-   [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Untested)
-   [`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B) (Untested)
-   [`Qwen/Qwen2.5-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) (Untested)
-   [`Qwen/Qwen2.5-14B`](https://huggingface.co/Qwen/Qwen2.5-14B) (Untested)
-   [`Qwen/Qwen2.5-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) (Untested)
-   [`Qwen/Qwen2.5-32B`](https://huggingface.co/Qwen/Qwen2.5-32B) (Untested)
-   [`Qwen/Qwen2.5-72B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Untested)
-   [`Qwen/Qwen2.5-72B`](https://huggingface.co/Qwen/Qwen2.5-7B) (Untested)

### Qwen3

This implementation should be compatible with all dense Qwen3 models. We mark version with explicit testing.

-   [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) (Tested ✅)
-   [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) (Tested ✅)
-   [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) (Tested ✅)
-   [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (Untested)
-   [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B) (Untested)
-   [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) (Untested)

---

## 🧐 Project Status & Performance

The primary goal of this project is educational: to demonstrate how cutting-edge architectures can be built and understood in a framework other than PyTorch/TensorFlow.

### Inference Performance
Currently, inference is **very slow**. This is an expected limitation of tinygrad's current Just-In-Time (JIT) compiler, which is optimized for tensors with **fixed shapes**. Autoregressive decoding, where the sequence length changes at each step, requires re-compiling the computation graph for every new token, hindering performance.

Here is a performance comparison for `LiquidAI/LFM2-350M` against the official PyTorch implementation on an AMD RX 6700 XT, including new quantization modes:

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

### Training Performance (LFM2)
In contrast, **training performance is surprisingly competitive**. Since the training loop uses fixed-size input batches (`batch_size`, `max_length`), the tinygrad JIT can compile highly optimized kernels once and reuse them.

In our testing on an **AMD RX 6700 XT**, a short LFM2 fine-tuning run (`max_length=512`, `batch_size=2`, `max_steps=100`) completed in approximately **4 minutes and 20 seconds**, utilizing around 10.1/12GB of VRAM. This demonstrates that tinygrad is a capable framework for training when tensor shapes are static.

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
To run standard text generation, use `run.py`. This script loads the pretrained LFM2-350M model by default.

```bash
# Run LFM2-350M (default)
python src/run.py

# Run Qwen3-0.6B
python src/run.py --model Qwen3 --model_id "Qwen/Qwen3-0.6B"
```

#### Inference with Quantization
You can enable NF4 or INT8 quantization to reduce memory usage with the `--quantize` flag.

```bash
# Run LFM2 with 4-bit NormalFloat quantization
python src/run.py --quantize nf4

# Run Qwen3 with 8-bit Integer quantization
python src/run.py --model Qwen3 --model_id "Qwen/Qwen3-0.6B" --quantize int8
```

#### Supervised Fine-Tuning (SFT) with LoRA
The `train_sft.py` script allows you to fine-tune on any conversational dataset from the Hugging Face Hub. LoRA is enabled by default for efficiency.

Here is an example command to run a short training job on the `mlabonne/FineTome-100k` dataset:

```bash
python src/train_sft.py \
    --model "LFM2"
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
This implementation has been carefully verified against the official HuggingFace transformers implementation. The debugging scripts (`src/debug_prefilling.py` and `src/debug_decoding.py`) perform a layer-by-layer comparison of hidden states and cache values, confirming that the tinygrad model produces numerically identical outputs.

---

## 🗺️ Roadmap

Our vision is to evolve this project into a comprehensive library for exploring and training diverse LLMs on a wide range of hardware, especially commodity GPUs.

-   [ ] **Expand Architectural Support:**
    -   **Goal:** Adapt more diverse model architectures from the Hugging Face ecosystem. The new `base_modeling.py` provides a strong foundation for rapid prototyping.
    -   **Targets:**
        -   **Mixture-of-Experts (MoE):** Implement models like Mixtral and Qwen2-MoE to explore sparse activation.
        -   **State Space Models (SSM):** Adapt architectures like Mamba or Jamba.
        -   **Vision Language Models (VLM):** Extend the framework to handle multi-modal inputs.

-   [ ] **Champion Alternative & Commodity GPUs:**
    -   **Goal:** Become a go-to library for running and fine-tuning LLMs on non-NVIDIA hardware by leveraging tinygrad's broad backend support (AMD, Intel, Apple Silicon).
    -   **Actions:**
        -   Provide extensive benchmarking results across different hardware.
        -   Develop hardware-specific optimization guides.
        -   Ensure all features (quantization, LoRA, etc.) are robustly tested across tinygrad's primary backends.

-   [ ] **Core Performance Enhancements:**
    -   **Improve Inference Speed:** Investigate workarounds for the dynamic shape problem, potentially by padding to fixed sequence length buckets or exploring future tinygrad JIT enhancements.
    -   **Optimize Paged Attention:** Refine the paged attention kernels once the core JIT issues are addressed.

---

## Acknowledgments
This project was heavily inspired by the following resources:
-   [kyegomez/LFM2](https://github.com/kyegomez/LFM2) for an early PyTorch implementation.
-   The official [Hugging Face Transformers implementation of LFM2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py).