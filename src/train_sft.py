# src/train_sft.py

import os
os.environ["VIZ"] = "1"

import json
import argparse
from typing import List
from tqdm import tqdm
import random
import time

# Third-party imports
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb 

# tinygrad imports
from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import GlobalCounters
from extra.lr_scheduler import OneCycleLR 

# Project imports
from model.fp16_lfm2_modeling import LFM2Config, LFM2ForCausalLM, load_from_hf
from model.lfm2_modeling import GroupedQueryAttention, LFM2ConvOperator, SwiGLU

# --- Dataset and Preprocessing ---

IGNORE_INDEX = -100

def data_generator(dataset, tokenizer, max_length, batch_size):
    """
    A custom data generator that processes, tokenizes, and batches conversational data,
    yielding tinygrad Tensors directly. It uses a buffering strategy to ensure full
    batches for TinyJit and trains the model only on the last assistant response in a conversation.
    """
    # Shuffle the dataset indices for each epoch
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    buffer_prompts = []
    buffer_full_texts = []

    for idx in indices:
        item = dataset[idx]

        # 1. Extract and convert conversation format from {from, value} to {role, content}
        conversation = item.get("conversations", [])
        if not conversation:
            continue

        messages = []
        for turn in conversation:
            # Map 'from' to 'role'. Assuming 'human' -> 'user' and 'gpt' -> 'assistant'
            role = "user" if turn.get("from") == "human" else "assistant"
            content = turn.get("value", "")
            messages.append({"role": role, "content": content})

        # 2. We only train on the last assistant response. Ensure the conversation ends with one.
        if not messages or messages[-1]["role"] != "assistant":
            continue

        # 3. Create the full text and the prompt text for masking purposes.
        # The full text is the entire conversation, which becomes the model's input.
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # The full conversation is provided
        )
        
        # The "prompt" text is the conversation *up to* the final response, with a
        # generation prompt added. This tells the model to start generating.
        # We tokenize this separately to find out how many initial tokens to mask in the labels.
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        buffer_full_texts.append(full_text)
        buffer_prompts.append(prompt_text)

        # 4. When the buffer is full, process and yield a batch
        if len(buffer_full_texts) == batch_size:
            # Batch tokenize the full texts with padding and truncation
            full_tokenized = tokenizer(
                buffer_full_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None,  # Return Python lists
            )

            # Batch tokenize prompts to get their lengths for masking
            prompt_tokenized = tokenizer(
                buffer_prompts,
                max_length=max_length,
                truncation=True,
                return_tensors=None,
            )

            input_ids_batch = full_tokenized['input_ids']
            labels_batch = []

            # Create labels for each item in the batch
            for j in range(batch_size):
                prompt_len = len(prompt_tokenized['input_ids'][j])
                
                # Start with a copy of the input_ids
                labels = list(input_ids_batch[j])
                
                # Mask the prompt part so loss is not calculated on it
                labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
                
                # Mask padding tokens in the labels as well
                for k in range(len(labels)):
                    if input_ids_batch[j][k] == tokenizer.pad_token_id:
                        labels[k] = IGNORE_INDEX
                
                labels_batch.append(labels)
                
            # Yield a batch of tinygrad tensors
            yield (
                Tensor(input_ids_batch, dtype=dtypes.int32, device=Device.DEFAULT),
                Tensor(labels_batch, dtype=dtypes.int32, device=Device.DEFAULT)
            )

            # Clear buffers for the next batch
            buffer_prompts = []
            buffer_full_texts = []

def estimate_mfu_flops(model: LFM2ForCausalLM, batch_size: int, seq_len: int) -> int:
    """
    Estimates the FLOPS for a forward/backward pass of the LFM2 model.
    """
    config = model.config
    B, S, H, I, K = batch_size, seq_len, config.hidden_size, config.intermediate_size, config.conv_kernel_size

    fwd_flops = 0
    
    # Iterate through each decoder layer
    for layer in model.model.layers:
        # 1. Shared MLP (SwiGLU) part
        # w1, w3, w2 projections
        fwd_flops += 6 * B * S * H * I

        # 2. Operator part (Attention or Convolution)
        if isinstance(layer.operator, GroupedQueryAttention):
            # Q, K, V, O projections (simplified as 4 * H*H)
            fwd_flops += 4 * B * S * H * H
            # Attention Score (Q@K^T) and Output (@V)
            # This is the sequence-length dependent part
            fwd_flops += 4 * B * (S**2) * H
        elif isinstance(layer.operator, LFM2ConvOperator):
            # in_proj (H -> 3H) and out_proj (H -> H)
            fwd_flops += (2 * B * S * H * (3 * H)) + (2 * B * S * H * H)
            # Depthwise conv (cheap)
            fwd_flops += 2 * B * H * S * K
    
    # Final layer norm and lm_head
    fwd_flops += 2 * B * S * config.vocab_size * H

    # Backward pass is ~2x forward pass
    return 3 * fwd_flops


def main(args):
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    print(f"--- Starting SFT Training for {args.model_id} on {Device.DEFAULT} ---")

    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_path = hf_hub_download(repo_id=args.model_id, filename="config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = LFM2Config.from_hf_config(config_dict)
    
    model = LFM2ForCausalLM(config)
    load_from_hf(model, args.model_id)
    
    print("\n--- Model & MFU Setup ---")
    if args.device_peak_flops > 0:
        total_params = sum(p.numel() for p in get_parameters(model))
        flops_per_step = estimate_mfu_flops(model, args.batch_size, args.max_length)
        print(f"  Total Parameters: {total_params / 1e6:.2f}M")
        print(f"  Sequence Length: {args.max_length}, Batch Size: {args.batch_size}")
        print(f"  Estimated FLOPS per step (arch-aware): {flops_per_step / 1e12:.2f} TFLOPS")
        print(f"  Provided Peak Hardware FLOPS: {args.device_peak_flops} TFLOPS")
    else:
        flops_per_step = 0
        print("  Skipping MFU calculation (device_peak_flops not provided).")

    for p in get_parameters(model):
        p.requires_grad = True

    print(f"\nLoading dataset '{args.dataset_id}'...")
    dataset = load_dataset(args.dataset_id, split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))

    params = get_parameters(model)
    for p in params:
        p.requires_grad = True

    # 3. Setup Optimizer and JIT'd Training Step
    optim = AdamW(params, lr=args.learning_rate) # LR here is just a placeholder, OneCycleLR will manage it
    
    # # --- MODIFIED: Use OneCycleLR ---
    # lr_scheduler = OneCycleLR(
    #     optim,
    #     max_lr=args.learning_rate,
    #     total_steps=args.max_steps,
    #     div_factor=10,          # Initial LR will be max_lr / 10
    #     final_div_factor=100,  # Final LR will be initial_lr / 100
    #     pct_start=0.1           # 10% of steps are for warmup
    # )

    @TinyJit
    def train_step(input_ids: Tensor, labels: Tensor):
        optim.zero_grad()
        
        output = model(input_ids, labels=labels)
        loss = output.loss
        loss.cast(dtypes.float32).backward()
        
        total_norm = Tensor(0.0, dtype=dtypes.float32, device=optim.params[0].device)
        for p in optim.params:
            if p.grad is not None:
                total_norm += p.grad.float().square().sum()
        total_norm = total_norm.sqrt().contiguous()
        for p in optim.params:
            if p.grad is not None:
                p.grad = p.grad * (args.gradient_clipping_norm / (total_norm + 1e-6)).clamp(max_=1.0)

        optim.step()
        # lr_scheduler.step()
        loss, out_lr, grad_norm_cpu = loss.detach().to("CPU"), optim.lr.to("CPU"), total_norm.detach().to("CPU")
        Tensor.realize(loss, out_lr, grad_norm_cpu)
        return loss, out_lr.item(), grad_norm_cpu

    # 4. Training Loop (no changes needed here)
    print("\n--- Starting Training ---")
    train_iterator = iter(data_generator(dataset, tokenizer, args.max_length, args.batch_size))
    pbar = tqdm(range(args.max_steps), desc="Training")
    
    step_times = []
    warmup_steps = 5 

    for step in pbar:
        with Tensor.train():
            try:
                input_ids, labels = next(train_iterator)
            except StopIteration:
                print("\nEpoch finished. Re-shuffling and creating new data generator...")
                train_iterator = iter(data_generator(dataset, tokenizer, args.max_length, args.batch_size))
                input_ids, labels = next(train_iterator)

            start_time = time.perf_counter()
            GlobalCounters.reset()
            loss, lr, grad_norm = train_step(input_ids, labels)
            end_time = time.perf_counter()
            
            step_time = end_time - start_time
            if step >= warmup_steps:
                step_times.append(step_time)

            mfu_str = "N/A"
            achieved_tflops = 0.0
            mfu = 0.0
            if flops_per_step > 0 and len(step_times) > 0:
                avg_step_time = sum(step_times) / len(step_times)
                achieved_tflops = flops_per_step / avg_step_time / 1e12
                mfu = achieved_tflops / args.device_peak_flops
                mfu_str = f"{mfu:.2%}"
            
            loss_val = loss.item()
            grad_norm_val = grad_norm.item()
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}", 
                "lr":  f"{lr:.0e}".replace("e-0", "e-"),
                "time": f"{step_time*1000:.2f}ms",
                "MFU": mfu_str
            })

            if args.use_wandb:
                wandb.log({
                    "train/loss": loss_val,
                    "train/learning_rate": lr,
                    "perf/step_time_ms": step_time * 1000,
                    "perf/grad_norm": grad_norm_val,
                    "perf/mfu": mfu,
                    "perf/achieved_tflops": achieved_tflops,
                }, step=step)


    print("\n--- Training Complete ---")
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with LFM2 on tinygrad")
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-350M", help="Hugging Face model repository ID")
    parser.add_argument("--dataset_id", type=str, default="mlabonne/FineTome-100k", help="Hugging Face dataset ID for SFT")
    parser.add_argument("--max_length", type=int, default=512, help="Fixed sequence length for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Optimizer learning rate")
    parser.add_argument("--gradient_clipping_norm", type=float, default=1.0, help="Optimizer gradient clipping norm")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of training steps")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use from the dataset (for quick tests)")
    parser.add_argument("--device_peak_flops", type=float, default=-1.0, help="Peak FP16/BF16 TFLOPS of the training device. E.g., A100: 312, RTX 4090: 330, RX 6700XT: ~23. If -1, MFU is not calculated.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="lfm2-tinygrad-sft", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=f"sft-run-{int(time.time())}", help="Wandb run name")
    
    args = parser.parse_args()
    main(args)