# src/train_sft.py

import json
import argparse
from typing import List
from tqdm import tqdm
import random

# Third-party imports
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from datasets import load_dataset

# tinygrad imports
from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import GlobalCounters
from extra.lr_scheduler import CosineAnnealingLRWithWarmup

# Project imports
from model.fp16_lfm2_modeling import LFM2Config, LFM2ForCausalLM, load_from_hf

# --- Dataset and Preprocessing ---

# Alpaca prompt template
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
IGNORE_INDEX = -100

def data_generator(dataset, tokenizer, max_length, batch_size):
    """
    A custom data generator that processes, tokenizes, and batches data,
    yielding tinygrad Tensors directly.
    """
    # Shuffle the dataset indices for each epoch
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        # Skip the last batch if it's smaller than the batch size, as TinyJit requires fixed shapes
        if len(batch_indices) < batch_size:
            continue

        batch_data = dataset[batch_indices]
        
        prompts = []
        full_texts = []
        
        for j in range(batch_size):
            item = {key: val[j] for key, val in batch_data.items()}
            
            prompt_input = item.get("input", "")
            prompt = PROMPT_TEMPLATE.format(instruction=item["instruction"], input=prompt_input)
            full_text = prompt + item["output"] + tokenizer.eos_token
            
            prompts.append(prompt)
            full_texts.append(full_text)

        # Batch tokenize the full texts with padding and truncation
        full_tokenized = tokenizer(
            full_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None, # Return Python lists
        )

        # Batch tokenize prompts to get their lengths for masking
        prompt_tokenized = tokenizer(
            prompts,
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
            
            # Mask the prompt part
            labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
            
            # Mask padding tokens in the labels
            for k in range(len(labels)):
                if input_ids_batch[j][k] == tokenizer.pad_token_id:
                    labels[k] = IGNORE_INDEX
            
            labels_batch.append(labels)
            
        # Yield a batch of tinygrad tensors
        yield (
            Tensor(input_ids_batch, dtype=dtypes.int32, device=Device.DEFAULT),
            Tensor(labels_batch, dtype=dtypes.int32, device=Device.DEFAULT)
        )

# --- Training ---

def main(args):
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
    
    # Set model parameters to require gradients
    for p in get_parameters(model):
        p.requires_grad = True

    # 2. Load and Prepare Dataset
    print(f"Loading dataset '{args.dataset_id}'...")
    dataset = load_dataset(args.dataset_id, split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))

    params = get_parameters(model)
    for p in params:
        p.requires_grad = True

    # 3. Setup Optimizer and JIT'd Training Step
    optim = AdamW(params, lr=args.learning_rate)

    lr_scheduler = CosineAnnealingLRWithWarmup(optim, base_lr=args.learning_rate, end_lr=0, warmup_steps=10, decay_steps=90) 

    @TinyJit
    def train_step(input_ids: Tensor, labels: Tensor):
        Tensor.training = True
        optim.zero_grad()
        
        output = model(input_ids, labels=labels)
        loss = output.loss
        loss.cast(dtypes.float32).backward()
        
        total_norm = Tensor(0.0, dtype=dtypes.float32, device=optim.params[0].device)
        for p in optim.params:
            total_norm += p.grad.float().square().sum()
        total_norm = total_norm.sqrt().contiguous()
        for p in optim.params:
            p.grad = p.grad * (args.gradient_clipping_norm / (total_norm + 1e-6)).clamp(max_=1.0)

        optim.step()
        lr_scheduler.step()
        lr = optim.lr

        return loss.realize(lr) # Realize loss for item() call and graph execution

    # 4. Training Loop
    print("\n--- Starting Training ---")
    train_iterator = iter(data_generator(dataset, tokenizer, args.max_length, args.batch_size))
    pbar = tqdm(range(args.max_steps), desc="Training")
    
    for step in pbar:
        try:
            input_ids, labels = next(train_iterator)
        except StopIteration:
            print("\nEpoch finished. Re-shuffling and creating new data generator...")
            train_iterator = iter(data_generator(dataset, tokenizer, args.max_length, args.batch_size))
            input_ids, labels = next(train_iterator)

        GlobalCounters.reset()
        loss = train_step(input_ids, labels)
        
        loss_val = loss.item()
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    print("\n--- Training Complete ---")
    # TODO: Add model saving logic here if desired

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with LFM2 on tinygrad")
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-350M", help="Hugging Face model repository ID")
    parser.add_argument("--dataset_id", type=str, default="tatsu-lab/alpaca", help="Hugging Face dataset ID for SFT")
    parser.add_argument("--max_length", type=int, default=512, help="Fixed sequence length for training")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Optimizer learning rate")
    parser.add_argument("--gradient_clipping_norm", type=float, default=1.0, help="Optimizer gradient clipping norm")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of training steps")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use from the dataset (for quick tests)")
    args = parser.parse_args()
    main(args)