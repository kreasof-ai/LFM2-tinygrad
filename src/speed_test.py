import sys

print(sys.path)

import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# --- tinygrad imports ---
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv

# --- Import your implementations ---
from model import paged_lfm2_modeling as paged_tg, lfm2_modeling as naive_tg


# --- Test Configuration ---
REPO_ID = "LiquidAI/LFM2-350M"
PROMPT = "The secret to a long and happy life is"
MAX_NEW_TOKENS = 128
# Use greedy decoding for a fair speed comparison (no random sampling)
TEMPERATURE = 0.0

# For reproducible tests if needed
if getenv("SEED"):
    Tensor.manual_seed(getenv("SEED"))

def run_huggingface_test(tokenizer):
    """
    Tests the inference speed of the Hugging Face reference model.
    """
    print("\n--- 1. Testing Hugging Face (PyTorch) Reference ---")
    print("Loading model... (This might take a moment)")
    
    # Load model to GPU with float16 for a realistic performance comparison
    model_hf = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model_hf.device)

    print("Generating tokens...")
    # Warmup run
    _ = model_hf.generate(input_ids, max_new_tokens=5, do_sample=False)

    # Timed run
    start_time = time.perf_counter()
    output = model_hf.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False, # Use greedy for speed test
        pad_token_id=tokenizer.eos_token_id # Suppress warning
    )
    
    # Ensure all GPU operations are finished before stopping the timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    generated_tokens = len(output[0]) - len(input_ids[0])
    elapsed_time = end_time - start_time
    tokens_per_sec = generated_tokens / elapsed_time

    print(f"Generated Text Sample: {tokenizer.decode(output[0, -generated_tokens:])[:80]}...")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_sec:.2f} tok/s")
    
    # Clean up GPU memory
    del model_hf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return elapsed_time, tokens_per_sec

def run_naive_tinygrad_test(tokenizer, config):
    """
    Tests the inference speed of the naive tinygrad implementation.
    """
    print("\n--- 2. Testing Naive tinygrad Implementation ---")
    print("Loading model...")
    model = naive_tg.LFM2ForCausalLM(config)
    naive_tg.load_from_hf(model, REPO_ID)

    print("Generating tokens...")
    # Warmup run is implicitly handled by the first few tokens of the main run
    
    start_time = time.perf_counter()
    
    # The generate function handles everything
    _ = naive_tg.generate(
        model,
        tokenizer,
        PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    tokens_per_sec = MAX_NEW_TOKENS / elapsed_time
    
    print(f"\nTime taken: {elapsed_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_sec:.2f} tok/s")
    
    del model
    return elapsed_time, tokens_per_sec

def run_paged_tinygrad_test(tokenizer, config):
    """
    Tests the inference speed of the paged attention tinygrad implementation.
    """
    print("\n--- 3. Testing tinygrad with Paged Attention ---")
    print("Loading model...")
    model = paged_tg.LFM2ForCausalLM(config)
    paged_tg.load_from_hf(model, REPO_ID)
    
    print("Generating tokens...")
    # Warmup run is implicitly handled by the first few tokens of the main run
    
    start_time = time.perf_counter()
    
    # The generate function handles allocation, generation, and freeing
    _ = paged_tg.generate(
        model,
        tokenizer,
        PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    tokens_per_sec = MAX_NEW_TOKENS / elapsed_time

    print(f"\nTime taken: {elapsed_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_sec:.2f} tok/s")
    
    del model
    return elapsed_time, tokens_per_sec


if __name__ == "__main__":
    print(f"Starting LFM2 Inference Speed Test")
    print(f"Prompt: '{PROMPT}'")
    print(f"Tokens to generate: {MAX_NEW_TOKENS}")
    print(f"tinygrad Device: {Device.DEFAULT}")
    print("-" * 50)

    # 1. Load shared resources (tokenizer, configs)
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    
    config_naive = naive_tg.LFM2Config.from_hf_config(config_dict)
    config_paged = paged_tg.LFM2Config.from_hf_config(config_dict)

    results = {}

    # 2. Run the tests
    results["huggingface"] = run_huggingface_test(tokenizer)
    results["naive_tinygrad"] = run_naive_tinygrad_test(tokenizer, config_naive)
    results["paged_tinygrad"] = run_paged_tinygrad_test(tokenizer, config_paged)
    
    # 3. Print final summary
    print("\n\n" + "=" * 50)
    print("           INFERENCE SPEED TEST SUMMARY")
    print("=" * 50)
    print(f"{'Implementation':<25} | {'Time Taken (s)':<15} | {'Tokens/sec':<10}")
    print("-" * 50)
    
    hf_time, hf_tps = results["huggingface"]
    print(f"{'Hugging Face (PyTorch)':<25} | {hf_time:<15.4f} | {hf_tps:<10.2f}")
    
    naive_time, naive_tps = results["naive_tinygrad"]
    print(f"{'Naive tinygrad':<25} | {naive_time:<15.4f} | {naive_tps:<10.2f}")

    paged_time, paged_tps = results["paged_tinygrad"]
    print(f"{'Paged tinygrad':<25} | {paged_time:<15.4f} | {paged_tps:<10.2f}")
    print("=" * 50)