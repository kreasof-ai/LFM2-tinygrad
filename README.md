<p align="center">
	<picture>
		<source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/5ea13eaf-38b9-4c52-85a4-eed2eaa99d4d">
		<img alt="openformer-logo" src="https://github.com/user-attachments/assets/5ea13eaf-38b9-4c52-85a4-eed2eaa99d4d" width=50%>
	</picture>
</p>
<h3 align="center">
A hackable library for running and fine-tuning modern transformer models on commodity and alternative GPUs, powered by <a href="https://github.com/tinygrad/tinygrad">tinygrad</a>
</h3>

---

The deep learning ecosystem is heavily dominated by a few major players and proprietary hardware vendors, making it difficult for developers and researchers to innovate on non-NVIDIA GPUs. **OpenFormer** is an open-source initiative by **[Kreasof AI](https://kreasof.my.id)** to democratize access to large language models.

Built on a unified, extensible architecture, OpenFormer liberates you to train, fine-tune, and run state-of-the-art transformer models on a wide range of hardware, including mainstream GPU from **AMD, Intel, and Apple Silicon GPUs**, by leveraging the power and simplicity of tinygrad.

---

## 🎯 Our Mission

*   ✅ **Democratize Access:** To be the go-to library for running and fine-tuning LLMs on commodity hardware. If it runs tinygrad, it can run OpenFormer.
*   ✅ **Simplify Complexity:** Provide a clean, understandable, and "hackable" from-scratch implementation of modern LLM architectures, built on a powerful `base_modeling` core.
*   ✅ **Champion Education:** Serve as a transparent, numerically verified educational tool for students, researchers, and developers to understand how LLMs work under the hood.

---

## 🌟 Key Features

-   **Modular, Extensible Core**: Built on a modular `base_modeling.py` that simplifies the adaptation of new Hugging Face models, often in just a few lines of code.
-   **Broad Architectural Support**: Ready-to-use, from-scratch implementations for diverse model families including **LFM2**, **Qwen**, and **Gemma 3**.
-   **Verified for Correctness**: Rigorously tested against official Hugging Face Transformers implementations to ensure numerically identical outputs.
-   **Built-in Fine-Tuning with LoRA**: A complete Supervised Fine-Tuning (SFT) script (`src/train_sft.py`) with integrated **Low-Rank Adaptation (LoRA)** support for efficient, low-memory training.
-   **Quantization Ready**: Out-of-the-box support for **INT8** and **NormalFloat4 (NF4)** quantization to reduce memory footprint during inference.
-   **Advanced Attention Mechanisms**: Includes experimental implementations of **Paged Attention** (for efficient KV cache management) and **Flash Attention**.

---

## ✅ Supported Architectures

Thanks to our flexible configuration loader, OpenFormer supports these model families:

### LFM2
-   `LiquidAI/LFM2-350M` (Default)
-   `LiquidAI/LFM2-700M`
-   `LiquidAI/LFM2-1.2B`
-   `LiquidAI/LFM2-2.6B`

### Qwen2 & Qwen2.5
This implementation is compatible with all dense Qwen2 and Qwen2.5 models.
-   `Qwen/Qwen2-0.5B-Instruct` (Tested ✅)
-   `Qwen/Qwen2.5-0.5B-Instruct` (Tested ✅)
-   And other variants (`1.5B`, `7B`, etc.)

### Qwen3
This implementation is compatible with all dense Qwen3 models.
-   `Qwen/Qwen3-0.6B` (Tested ✅)
-   `Qwen/Qwen3-1.7B` (Tested ✅)
-   `Qwen/Qwen3-4B` (Tested ✅)
-   And other variants (`8B`, `14B`, etc.)

### Gemma 3
This implementation supports text-only Gemma 3 models.
-   `google/gemma-3-270m-it` (Tested ✅)
-   `google/gemma-3-1b-it` (Tested ✅)

---

## 🧐 Performance: The tinygrad JIT Trade-off

OpenFormer's performance profile directly reflects the strengths and current limitations of the tinygrad JIT (Just-In-Time) compiler.

### Inference: Slow but Steady
Currently, autoregressive decoding (inference) is **slow**. tinygrad's JIT is optimized for tensors with **fixed shapes**. Because the sequence length changes at each generation step, the computation graph must be re-compiled for every new token, creating significant overhead.

Here is a performance comparison for `LiquidAI/LFM2-350M` against PyTorch on an **AMD RX 6700 XT**:

| Implementation                 | Time Taken (s) for 64 tokens | Tokens/sec |
| ------------------------------ | ---------------------------- | ---------- |
| Hugging Face (PyTorch)         | 2.6467                       | 24.18      |
| **OpenFormer (FP32)**          | **65.3724**                  | **0.98**   |
| **OpenFormer (FP16)**          | **71.8813**                  | **0.89**   |
| **OpenFormer (INT8)**          | **63.3681**                  | **1.01**   |
| OpenFormer (Paged, FP32)       | 77.2624                      | 0.83       |

*As shown, the primary bottleneck is kernel compilation, not memory access or data type precision.*

### Training: Surprisingly Fast
In contrast, **training performance is highly competitive**. The training loop uses fixed-size input batches (`batch_size`, `max_length`), allowing the tinygrad JIT to compile highly optimized kernels once and reuse them.

On an **AMD RX 6700 XT**, a short LFM2 LoRA fine-tuning run (`max_length=512`, `batch_size=2`, `max_steps=100`) completed in approximately **4 minutes and 20 seconds**, utilizing ~10.1/12GB of VRAM. This proves that tinygrad is a powerful and viable framework for training on non-NVIDIA hardware.

<img alt="Screenshot of a successful training run" src="https://github.com/user-attachments/assets/861afba5-ab6e-4dff-ab50-424e5cb0a56d" />

---

## 🚀 Getting Started

| Notebook | Link |
|----------|------|
| Kaggle   | <a href="https://www.kaggle.com/code/akbar2habibullah/openformer-kaggle" target="_blank"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" /></a> |

> Note: The Kaggle notebook provides a ready-to-use T4 GPU environment. Google Colab is not supported at this time due to driver issues.

### 1. Installation
Clone the repository and install the required dependencies.

```bash
git clone https://github.com/kreasof-ai/OpenFormer.git
cd OpenFormer
pip install tinygrad torch transformers huggingface_hub safetensors tqdm datasets wandb
```

### 2. Usage Examples

#### Inference
To run standard text generation, use `run.py`.

```bash
# Run LFM2-350M (default)
python src/run.py

# Run a different model, like Qwen3-0.6B
python src/run.py --model Qwen3 --model_id "Qwen/Qwen3-0.6B"
```

#### Inference with Quantization
Enable NF4 or INT8 quantization to reduce memory usage with the `--quantize` flag.

```bash
# Run LFM2 with 4-bit NormalFloat quantization
python src/run.py --quantize nf4

# Run Qwen3 with 8-bit Integer quantization
python src/run.py --model Qwen3 --model_id "Qwen/Qwen3-0.6B" --quantize int8
```

#### Supervised Fine-Tuning (SFT) with LoRA
The `train_sft.py` script allows you to fine-tune any supported model on a conversational dataset from the Hugging Face Hub. LoRA is enabled by default for efficiency.

Here is an example command to fine-tune `LFM2-350M` on the `mlabonne/FineTome-100k` dataset:

```bash
python src/train_sft.py \
    --model "LFM2" \
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
    --wandb_project "openformer-sft"
```
*   To perform a full fine-tune (instead of LoRA), remove the `--use_lora` flag.
*   The script automatically handles data processing, masking labels for prompts, and logging to Weights & Biases.

---

## 💾 Saving & Uploading Fine-Tuned Models

After fine-tuning, you can easily save your model and upload it to the Hugging Face Hub. The `save_pretrained` method handles dequantization, weight name mapping, and copies all necessary configuration files from the original repository.

Here's how to save a fine-tuned model and push it to the Hub:

```python
# Assuming 'model' is your fine-tuned OpenFormer model instance
# from train_sft.py or loaded otherwise.

# 1. Save locally
model.save_pretrained("./my-finetuned-lfm2")

# 2. Save locally AND upload to the Hub
# Make sure you are logged in via `huggingface-cli login`
# model.save_pretrained(
#     save_directory="./my-finetuned-lfm2-hub",
#     repo_id="your-username/my-finetuned-lfm2"
# )
```

---

## 🗺️ Roadmap

Our vision is to evolve OpenFormer into a comprehensive library for training and deploying diverse LLMs on a wide range of hardware.

-   [ ] **Champion Alternative & Commodity GPUs:**
    -   **Goal:** Become the premier library for LLMs on non-NVIDIA hardware by leveraging tinygrad's broad backend support (AMD, Intel, Apple Silicon).
    -   **Actions:** Provide extensive benchmarking, develop hardware-specific optimization guides, and ensure all features are robustly tested across backends.

-   [ ] **Expand Architectural Support:**
    -   **Goal:** Rapidly adapt more diverse model architectures from the Hugging Face ecosystem using our proven base modeling classes.
    -   **Targets:** Mixture-of-Experts (MoE), State Space Models (SSM) like Mamba, and Vision Language Models (VLM).

-   [ ] **Core Performance Enhancements:**
    -   **Improve Inference Speed:** Investigate workarounds for the dynamic shape problem, potentially by padding to fixed sequence length buckets or contributing to future tinygrad JIT enhancements.
    -   **Optimize Kernels:** Refine and optimize experimental kernels like paged and flash attention.

---

## ❤️ Contributing

We believe in the power of open source to challenge the status quo. Contributions are welcome! Whether it's adding a new model, improving performance, or fixing a bug, please feel free to open an issue or submit a pull request.

---

## Acknowledgments
This project was heavily inspired by the official Hugging Face Transformers library and the innovative work of the tinygrad community. We also acknowledge the original PyTorch implementation of LFM2 by [kyegomez/LFM2](https://github.com/kyegomez/LFM2).
