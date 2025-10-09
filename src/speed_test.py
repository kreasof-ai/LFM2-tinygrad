# /src/speed_test.py (Unified)

import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# --- tinygrad imports ---
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

# --- Import the new unified implementation ---
from model import lfm2_modeling

# --- Test Configuration ---
REPO_ID = "LiquidAI/LFM2-350M"
PROMPT = "The secret to a long and happy life is"
MAX_NEW_TOKENS = 64
# Use greedy decoding for a fair speed comparison (no random sampling)
TEMPERATURE = 0.0

# For reproducible tests if needed
if getenv("SEED"):
    Tensor.manual_seed(getenv("SEED"))

def run_huggingface_test(tokenizer):
    """
    Tests the inference speed of the Hugging Face reference model (FP16).
    """
    print("\n--- 1. Testing Hugging Face (PyTorch) Reference ---")
    print("Loading model... (This might take a moment)")
    
    model_hf = AutoModelForCausalLM.from_pretrained(
        REPO_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_tensors="pt", tokenize=True,
    ).to(model_hf.device)

    print("Generating tokens...")
    # Warmup run
    _ = model_hf.generate(input_ids, max_new_tokens=5, do_sample=False)

    # Timed run
    start_time = time.perf_counter()
    output = model_hf.generate(
        input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    generated_tokens = len(output[0]) - len(input_ids[0])
    elapsed_time = end_time - start_time
    tokens_per_sec = generated_tokens / elapsed_time

    print(f"Generated Text Sample: {tokenizer.decode(output[0, -generated_tokens:])}...")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_sec:.2f} tok/s")
    
    del model_hf
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return elapsed_time, tokens_per_sec

def run_tinygrad_test(name: str, tokenizer, config: lfm2_modeling.LFM2Config):
    """
    A unified function to test any tinygrad configuration.
    The behavior (paged vs. standard, FP16 vs. FP32) is controlled by the config object.
    """
    print(f"\n--- Testing: {name} ---")
    print(f"  Config: Paged Attention={'ON' if config.use_paged_attention else 'OFF'}, DType={config.dtype}")
    print("Loading model...")
    model = lfm2_modeling.LFM2ForCausalLM(config)
    lfm2_modeling.load_from_hf(model, REPO_ID)

    print("Generating tokens...")
    # The unified generate function automatically handles the correct execution path
    # based on the flags set in the config object.
    start_time = time.perf_counter()
    _ = lfm2_modeling.generate(
        model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
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

    # 1. Load shared resources
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path) as f: config_dict = json.load(f)

    # 2. Create all necessary configurations
    config_std_fp32 = lfm2_modeling.LFM2Config.from_hf_config(config_dict)
    config_std_fp32.use_paged_attention = False
    config_std_fp32.dtype = dtypes.float32

    config_std_fp16 = lfm2_modeling.LFM2Config.from_hf_config(config_dict)
    config_std_fp16.use_paged_attention = False
    config_std_fp16.dtype = dtypes.float16

    config_paged_fp32 = lfm2_modeling.LFM2Config.from_hf_config(config_dict)
    config_paged_fp32.use_paged_attention = True
    config_paged_fp32.dtype = dtypes.float32
    
    config_paged_fp16 = lfm2_modeling.LFM2Config.from_hf_config(config_dict)
    config_paged_fp16.use_paged_attention = True
    config_paged_fp16.dtype = dtypes.float16

    # 3. Define the test battery
    tests_to_run = [
        ("huggingface", run_huggingface_test, (tokenizer,)),
        ("std_fp32", run_tinygrad_test, ("Standard tinygrad (FP32)", tokenizer, config_std_fp32)),
        ("std_fp16", run_tinygrad_test, ("Standard tinygrad (FP16)", tokenizer, config_std_fp16)),
        ("paged_fp32", run_tinygrad_test, ("Paged tinygrad (FP32)", tokenizer, config_paged_fp32)),
        ("paged_fp16", run_tinygrad_test, ("Paged tinygrad (FP16)", tokenizer, config_paged_fp16)),
    ]
    
    results = {}
    for key, func, args in tests_to_run:
        results[key] = func(*args)
    
    # 4. Print final summary
    print("\n\n" + "=" * 60)
    print("           INFERENCE SPEED TEST SUMMARY")
    print("=" * 60)
    print(f"{'Implementation':<30} | {'Time Taken (s)':<15} | {'Tokens/sec':<10}")
    print("-" * 60)
    
    hf_time, hf_tps = results["huggingface"]
    print(f"{'Hugging Face (PyTorch)':<30} | {hf_time:<15.4f} | {hf_tps:<10.2f}")
    
    s32_time, s32_tps = results["std_fp32"]
    print(f"{'Standard tinygrad (FP32)':<30} | {s32_time:<15.4f} | {s32_tps:<10.2f}")

    s16_time, s16_tps = results["std_fp16"]
    print(f"{'Standard tinygrad (FP16)':<30} | {s16_time:<15.4f} | {s16_tps:<10.2f}")

    p32_time, p32_tps = results["paged_fp32"]
    print(f"{'Paged tinygrad (FP32)':<30} | {p32_time:<15.4f} | {p32_tps:<10.2f}")

    p16_time, p16_tps = results["paged_fp16"]
    print(f"{'Paged tinygrad (FP16)':<30} | {p16_time:<15.4f} | {p16_tps:<10.2f}")
    print("=" * 60)