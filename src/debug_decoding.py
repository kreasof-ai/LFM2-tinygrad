# /src/debug_decoding.py

import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinygrad import Tensor, dtypes

# Use the new unified model loading
from model import MODEL_MAP

# --- Comparison Helper ---
def compare_tensors(tg_tensor: Tensor, pt_tensor: torch.Tensor, name: str):
    """Compares a tinygrad tensor and a PyTorch tensor."""
    print(f"--- Comparing: {name} ---")
    tg_np = tg_tensor.numpy()
    pt_np = pt_tensor.detach().cpu().to(torch.float32).numpy()

    print(f"  Shapes: TG={tg_np.shape}, PT={pt_np.shape}")
    if tg_np.shape != pt_np.shape:
        print(f"  ❌ MISMATCH: Shapes are different!")
        return False

    print(f"  Means:  TG={tg_np.mean():.6f}, PT={pt_np.mean():.6f}")
    
    max_abs_diff = np.max(np.abs(tg_np - pt_np))
    print(f"  Max absolute difference: {max_abs_diff:.8f}")

    is_close = np.allclose(tg_np, pt_np, atol=1e-3, rtol=1e-3)
    if is_close: print("  ✅ MATCH: Tensors are numerically close.")
    else: print("  ❌ MISMATCH: Tensors are NOT close.")
    print("-" * (len(name) + 22))
    return is_close

def compare_caches_transformer(tg_cache: list, hf_cache, num_layers: int, name: str, current_seq_len: int):
    """Compares standard transformer KV caches."""
    print(f"\n--- Comparing Caches: {name} (Seq Len: {current_seq_len}) ---")
    all_match = True
    for i in range(num_layers):
        tg_k, tg_v = tg_cache[i]
        hf_k, hf_v = hf_cache[i]
        if not compare_tensors(tg_k, hf_k, f"Layer {i} Key Cache"): all_match = False
        if not compare_tensors(tg_v, hf_v, f"Layer {i} Value Cache"): all_match = False
        if not all_match:
            print(f"‼️ DIVERGENCE DETECTED IN CACHE AT LAYER {i} ‼️")
            return False
    print("--- Finished Comparing Caches ---")
    return all_match


# --- Main Comparison Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug the decoding stage of a model.")
    parser.add_argument("--model", type=str, default="LFM2", choices=MODEL_MAP.keys(), help="Model to debug.")
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-350M", help="Hugging Face model ID.")
    args = parser.parse_args()

    PROMPT = "The secret is"

    # 1. Load HF model
    model_hf = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # 2. Load tinygrad model
    print("\nLoading tinygrad model...")
    CausalLM = MODEL_MAP[args.model]
    model_tg = CausalLM.from_pretrained(args.model_id)

    # 3. Prepare inputs
    input_ids_pt = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    input_ids_tg = Tensor(input_ids_pt.numpy().astype(np.int32))
    prompt_len = input_ids_pt.shape[1]

    # --- 4. PREFILL STAGE COMPARISON ---
    print("\n--- Starting Prefill Stage Comparison ---")
    with torch.no_grad():
        pt_prefill_out = model_hf(input_ids_pt, use_cache=True)
    
    tg_prefill_out = model_tg(input_ids_tg, start_pos=0)
    
    logits_pt_prefill, cache_hf_prefill = pt_prefill_out.logits, pt_prefill_out.past_key_values
    logits_tg_prefill, cache_tg_prefill = tg_prefill_out.logits, tg_prefill_out.past_key_values
    
    if not compare_tensors(logits_tg_prefill[:, -1, :], logits_pt_prefill[:, -1, :], "Prefill - Final Logit"): exit()
    if not compare_caches_transformer(cache_tg_prefill, cache_hf_prefill, model_tg.config.num_hidden_layers, "Prefill Cache", prompt_len): exit()
    print("\n✅ Prefill stage matches perfectly!")

    # --- 5. DECODING STAGE COMPARISON ---
    print("\n--- Starting Decoding Stage Comparison ---")
    current_cache_hf = cache_hf_prefill
    current_cache_tg = cache_tg_prefill
    current_seq_len = prompt_len

    for i in range(5):
        print(f"\n>>>>>>>>>> DECODE STEP {i} <<<<<<<<<<")
        
        next_token_pt = torch.argmax(logits_pt_prefill[:, -1, :], dim=-1).unsqueeze(0)
        next_token_tg = Tensor([[logits_tg_prefill[0, -1].argmax().item()]], dtype=dtypes.int32)
        assert next_token_pt.item() == next_token_tg.item(), f"Token choice mismatch!"
        print(f"Generating with token ID: {next_token_pt.item()}")

        with torch.no_grad():
            pt_decode_out = model_hf(input_ids=next_token_pt, past_key_values=current_cache_hf, use_cache=True)
        
        tg_decode_out = model_tg(next_token_tg, past_states=current_cache_tg, start_pos=current_seq_len)
        
        logits_pt_decode, cache_hf_decode = pt_decode_out.logits, pt_decode_out.past_key_values
        logits_tg_decode, cache_tg_decode = tg_decode_out.logits, tg_decode_out.past_key_values
        
        if not compare_tensors(logits_tg_decode, logits_pt_decode, f"Decode Step {i} - Logits"): exit()
        if not compare_caches_transformer(cache_tg_decode, cache_hf_decode, model_tg.config.num_hidden_layers, f"Decode Step {i} - Cache", current_seq_len + 1): exit()
            
        logits_pt_prefill, current_cache_hf = logits_pt_decode, cache_hf_decode
        logits_tg_prefill, current_cache_tg = logits_tg_decode, cache_tg_decode
        current_seq_len += 1

    print("\n🎉🎉🎉 All checks passed! Prefill and 5 decode steps match perfectly. 🎉🎉🎉")