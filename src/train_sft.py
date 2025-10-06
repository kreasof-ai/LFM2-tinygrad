# src/train_sft.py

import json
import argparse
from tqdm import tqdm
import numpy as np

# Third-party imports
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from datasets import load_dataset
import torch # Using torch dataloader for convenience
from torch.utils.data import DataLoader, Dataset

# tinygrad imports
from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import GlobalCounters

# Project imports
from model.lfm2_modeling import LFM2Config, LFM2ForCausalLM, load_from_hf

# --- Dataset and Preprocessing ---

# Alpaca prompt template
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.IGNORE_INDEX = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the prompt
        if item.get("input", ""):
            prompt = PROMPT_TEMPLATE.format(instruction=item["instruction"], input=item["input"])
        else:
            prompt = PROMPT_TEMPLATE.format(instruction=item["instruction"], input="")
        
        full_text = prompt + item["output"]
        
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt)
        full_ids = self.tokenizer.encode(full_text + self.tokenizer.eos_token)
        
        # Create labels, masking out the prompt part
        labels = list(np.full(len(prompt_ids), self.IGNORE_INDEX)) + full_ids[len(prompt_ids):]
        
        # Pad/truncate
        input_ids = full_ids[:self.max_length]
        labels = labels[:self.max_length]
        
        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            labels = labels + [self.IGNORE_INDEX] * padding_len
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

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

    train_dataset = SFTDataset(dataset, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. Setup Optimizer and JIT'd Training Step
    optim = AdamW(get_parameters(model), lr=args.learning_rate)

    @TinyJit
    def train_step(input_ids: Tensor, labels: Tensor):
        Tensor.training = True
        optim.zero_grad()
        output = model(input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        optim.step()
        return loss

    # 4. Training Loop
    print("\n--- Starting Training ---")
    step = 0
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # Convert torch tensors to tinygrad tensors
            input_ids = Tensor(batch['input_ids'].numpy(), dtype=dtypes.int32, device=Device.DEFAULT)
            labels = Tensor(batch['labels'].numpy(), dtype=dtypes.int32, device=Device.DEFAULT)
            
            GlobalCounters.reset()
            loss = train_step(input_ids, labels)
            
            # Realize the loss to trigger computation and get its value
            loss_val = loss.item()
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
            step += 1
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            print(f"Reached max_steps ({args.max_steps}). Stopping training.")
            break

    print("\n--- Training Complete ---")
    # TODO: Add model saving logic here if desired

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with LFM2 on tinygrad")
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-350M", help="Hugging Face model repository ID")
    parser.add_argument("--dataset_id", type=str, default="tatsu-lab/alpaca", help="Hugging Face dataset ID for SFT")
    parser.add_argument("--max_length", type=int, default=512, help="Fixed sequence length for training")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Optimizer learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of training steps")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use from the dataset (for quick tests)")
    args = parser.parse_args()
    main(args)