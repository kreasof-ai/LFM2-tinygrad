# /src/debug_decoding.py

import json
import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinygrad import Tensor, dtypes

# Use the new unified model
from model.lfm2_modeling import LFM2ForCausalLM, LFM2Config, load_from_hf, hf_hub_download
from experimental.paged_attention import PagedKVCache # Needed for type checking in cache comparison

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

    is_close = np.allclose(tg_np, pt_np, atol=1e-4, rtol=1e-3)
    if is_close: print("  ✅ MATCH: Tensors are numerically close.")
    else: print("  ❌ MISMATCH: Tensors are NOT close.")
    print("-" * (len(name) + 22))
    return is_close

# --- Paged Cache Helper (only used in paged mode) ---
def reconstruct_from_paged_cache(paged_cache: PagedKVCache, batch_idx: int, seq_len: int):
    """Gathers from the physical cache to reconstruct the logical cache tensor."""
    if seq_len == 0:
        shape = (1, paged_cache.num_heads, 0, paged_cache.head_dim)
        return Tensor.empty(*shape), Tensor.empty(*shape)
    positions = Tensor.arange(seq_len).reshape(1, seq_len)
    batch_indices = Tensor([[batch_idx]]).expand(1, seq_len)    
    addrs = paged_cache.page_table.get_physical_addrs(batch_indices, positions).flatten()
    gather_indices = addrs.reshape(1, 1, -1, 1).expand(1, paged_cache.num_heads, -1, paged_cache.head_dim)
    k_logical = paged_cache.k_cache.gather(2, gather_indices).reshape(1, paged_cache.num_heads, seq_len, paged_cache.head_dim)
    v_logical = paged_cache.v_cache.gather(2, gather_indices).reshape(1, paged_cache.num_heads, seq_len, paged_cache.head_dim)
    return k_logical, v_logical

# --- Unified Cache Comparison ---
def compare_caches(tg_states: list, hf_cache, config: LFM2Config, name: str, batch_idx: int = 0, current_seq_len: int = 0) -> bool:
    """Compares tinygrad's caches (standard or paged) with the Hugging Face cache."""
    print(f"\n--- Comparing Caches: {name} (Seq Len: {current_seq_len}) ---")
    all_match = True
    for i in range(config.num_hidden_layers):
        is_attn = i in config.full_attn_idxs
        if is_attn:
            tg_cache_obj = tg_states[i]
            # Check the type to determine if we are in paged mode for this layer
            if isinstance(tg_cache_obj, PagedKVCache):
                tg_k, tg_v = reconstruct_from_paged_cache(tg_cache_obj, batch_idx, current_seq_len)
            else: # Standard mode
                tg_k, tg_v = tg_cache_obj
            
            hf_k, hf_v = hf_cache[i]
            if not compare_tensors(tg_k, hf_k, f"Layer {i} Key Cache"): all_match = False
            if not compare_tensors(tg_v, hf_v, f"Layer {i} Value Cache"): all_match = False
        else: # Convolution layer (same for both modes)
            if not compare_tensors(tg_states[i], hf_cache.conv_cache[i], f"Layer {i} Conv Cache"): all_match = False

        if not all_match:
            print(f"‼️ DIVERGENCE DETECTED IN CACHE AT LAYER {i} ‼️")
            return False
    print("--- Finished Comparing Caches ---")
    return all_match

# --- Main Comparison Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug the decoding (generation) stage of LFM2.")
    parser.add_argument("--paged", action="store_true", help="Run debugging in Paged Attention mode.")
    args = parser.parse_args()

    REPO_ID = "LiquidAI/LFM2-350M"
    PROMPT = "The secret is"

    # 1. Load HF model
    model_hf = AutoModelForCausalLM.from_pretrained(REPO_ID, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    # 2. Load tinygrad model
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path) as f: config_dict = json.load(f)
    config_tg = LFM2Config.from_hf_config(config_dict)

    if args.paged:
        print("\n*** RUNNING IN PAGED ATTENTION MODE ***\n")
        config_tg.use_paged_attention = True
    else:
        print("\n*** RUNNING IN STANDARD MODE ***\n")
    
    model_tg = LFM2ForCausalLM(config_tg)
    load_from_hf(model_tg, REPO_ID)

    # 3. Prepare inputs and paged-mode controller if needed
    input_ids_pt = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    input_ids_tg = Tensor(input_ids_pt.numpy().astype(np.int32))
    prompt_len = input_ids_pt.shape[1]

    controller = model_tg.page_table if args.paged else None
    batch_idx_int = -1

    try:
        if args.paged:
            model_tg.reset_request_state()
            batch_idx_int = controller.allocate()
            batch_idx_tensor = Tensor([batch_idx_int], dtype=dtypes.int32)
            controller.reserve(batch_idx_int, prompt_len + 10) # Reserve space for prompt + 5 steps

        # --- 4. PREFILL STAGE COMPARISON ---
        print("\n--- Starting Prefill Stage Comparison ---")
        with torch.no_grad():
            pt_prefill_out = model_hf(input_ids_pt, use_cache=True)
        
        if args.paged:
            tg_prefill_out = model_tg(input_ids_tg, start_pos=0, batch_idx=batch_idx_tensor, seq_lens=[prompt_len])
            cache_tg_prefill = model_tg.layer_caches
        else:
            tg_prefill_out = model_tg(input_ids_tg, start_pos=0)
            cache_tg_prefill = tg_prefill_out.past_key_values

        logits_pt_prefill, cache_hf_prefill = pt_prefill_out.logits, pt_prefill_out.past_key_values
        logits_tg_prefill = tg_prefill_out.logits
        
        if not compare_tensors(logits_tg_prefill[:, -1, :], logits_pt_prefill[:, -1, :], "Prefill - Final Logit"): exit()
        if not compare_caches(cache_tg_prefill, cache_hf_prefill, config_tg, "Prefill Cache", batch_idx_int, prompt_len): exit()
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
                pt_decode_out = model_hf(input_ids=next_token_pt, past_key_values=current_cache_hf, use_cache=True, position_ids=torch.tensor([[current_seq_len]]))
            
            if args.paged:
                tg_decode_out = model_tg(next_token_tg, start_pos=current_seq_len, batch_idx=batch_idx_tensor, seq_lens=[current_seq_len + 1])
                cache_tg_decode = model_tg.layer_caches
            else:
                tg_decode_out = model_tg(next_token_tg, past_states=current_cache_tg, start_pos=current_seq_len)
                cache_tg_decode = tg_decode_out.past_key_values

            logits_pt_decode, cache_hf_decode = pt_decode_out.logits, pt_decode_out.past_key_values
            logits_tg_decode = tg_decode_out.logits
            
            if not compare_tensors(logits_tg_decode, logits_pt_decode, f"Decode Step {i} - Logits"): exit()
            if not compare_caches(cache_tg_decode, cache_hf_decode, config_tg, f"Decode Step {i} - Cache", batch_idx_int, current_seq_len + 1): exit()
                
            logits_pt_prefill, current_cache_hf = logits_pt_decode, cache_hf_decode
            logits_tg_prefill, current_cache_tg = logits_tg_decode, cache_tg_decode
            current_seq_len += 1

        print("\n🎉🎉🎉 All checks passed! Prefill and 5 decode steps match perfectly. 🎉🎉🎉")

    finally:
        if args.paged and batch_idx_int != -1:
            controller.erase(batch_idx_int)
            print(f"\nErased batch slot {batch_idx_int}.")