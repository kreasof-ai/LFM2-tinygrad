# src/train_sft_hf.py

import os
import argparse
import time
import torch

# Third-party imports
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def main(args):
    print(f"--- Starting HF SFT Training for {args.model_id} on {'cuda' if torch.cuda.is_available() else 'cpu'} ---")

    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_fp16:
        print("  Using FP16 for training")
    else:
        print("  Using FP32 for training")

    model_kwargs = {"torch_dtype": torch.float16 if args.use_fp16 else torch.float32}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # 2. Load Dataset
    print(f"\nLoading dataset '{args.dataset_id}'...")
    dataset = load_dataset(args.dataset_id, split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))

    # 3. Configure LoRA (if enabled)
    peft_config = None
    if args.use_lora:
        print("\n--- LoRA is enabled ---")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"  Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha} to modules: {args.lora_target_modules}")
    else:
        print("\n--- Full finetuning enabled ---")


    # 4. Configure Training Arguments
    training_args = SFTConfig(
        output_dir="./hf_sft_results",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        max_grad_norm=args.gradient_clipping_norm,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        dataset_text_field="conversations", # The SFTTrainer can format chat templates from a column
        warmup_steps=10,
        logging_steps=1,
        save_strategy="no",
        fp16=args.use_fp16,
        max_length=args.max_length,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name,
    )

    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    # 6. Start Training
    print("\n--- Starting Training ---")
    train_result = trainer.train()

    # 7. Print Summary
    print("\n--- Training Complete ---")
    metrics = train_result.metrics
    print(f"  Time taken: {metrics['train_runtime']:.2f} seconds")
    print(f"  Total training loss: {metrics['train_loss']:.4f}")
    print(f"  Samples per second: {metrics['train_samples_per_second']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face SFT comparison script")
    # Model and Data
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-350M", help="Hugging Face model repository ID")
    parser.add_argument("--dataset_id", type=str, default="mlabonne/FineTome-100k", help="Hugging Face dataset ID for SFT")
    # Training Hyperparameters
    parser.add_argument("--max_length", type=int, default=512, help="Fixed sequence length for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--gradient_clipping_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples from dataset (for quick tests)")
    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj"], help="Module names to apply LoRA to")
    # Configuration Toggles
    parser.add_argument("--use_fp16", action="store_true", help="Enable FP16 training for lower memory usage")
    # Logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="lfm2-hf-sft", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=f"sft-run-hf-{int(time.time())}", help="Wandb run name")

    # Parse known arguments and ignore the rest
    args, unknown = parser.parse_known_args()
    main(args)