# /src/debug_prefilling.py

import json
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
    pt_np = pt_tensor.detach().cpu().numpy().astype(np.float32)

    print(f"  Shapes: TG={tg_np.shape}, PT={pt_np.shape}")
    if tg_np.shape != pt_np.shape:
        print("  ❌ MISMATCH: Shapes are different!")
        return False

    print(f"  Means:  TG={tg_np.mean():.6f}, PT={pt_np.mean():.6f}")
    
    max_abs_diff = np.max(np.abs(tg_np - pt_np))
    print(f"  Max absolute difference: {max_abs_diff:.6f}")

    is_close = np.allclose(tg_np, pt_np, atol=1e-3, rtol=1e-3)
    if is_close:
        print("  ✅ MATCH: Tensors are numerically close.")
    else:
        print("  ❌ MISMATCH: Tensors are NOT close.")
    print("-" * (len(name) + 22))
    return is_close

# --- Main Comparison Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug the prefilling stage of a model.")
    parser.add_argument("--model", type=str, default="LFM2", choices=MODEL_MAP.keys(), help="Model to debug.")
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-350M", help="Hugging Face model ID.")
    args = parser.parse_args()

    PROMPT = "The secret is"

    # 1. Load Hugging Face reference model
    print(f"Loading HF model: {args.model_id}...")
    model_hf = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # 2. Load tinygrad model
    print(f"\nLoading tinygrad model: {args.model} from {args.model_id}...")
    CausalLM = MODEL_MAP[args.model]
    model_tg = CausalLM.from_pretrained(args.model_id)

    # 3. Prepare identical inputs
    input_ids_pt = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    input_ids_tg = Tensor(input_ids_pt.numpy().astype(np.int32))
    seq_len = input_ids_pt.shape[1]

    # 4. Run full forward pass on Hugging Face model
    with torch.no_grad():
        pt_out = model_hf(input_ids_pt, output_hidden_states=True)
        logits_pt = pt_out.logits
        hidden_states_pt = pt_out.hidden_states
        position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        if args.model == "Qwen2" or args.model == "Qwen3" or args.model == "Gemma3":
            cos_pt, sin_pt = model_hf.model.rotary_emb(hidden_states_pt[0], position_ids)
        elif args.model == "LFM2":
            cos_pt, sin_pt = model_hf.model.pos_emb(hidden_states_pt[0], position_ids)

    # 5. Run tinygrad model
    print("\n--- Starting Step-by-Step tinygrad Comparison ---")
    tg_out = model_tg(input_ids_tg, output_hidden_states=True)

    logits_tg = tg_out.logits
    hidden_states_tg = tg_out.hidden_states

    # 6. Perform comparisons
    all_checks_passed = True
    
    # A. Compare initial embeddings
    if not compare_tensors(hidden_states_tg[0], hidden_states_pt[0], "Initial Embeddings"): all_checks_passed = False

    # B. Compare Layer Outputs
    if all_checks_passed:
        cos_tg_slice = model_tg.model.cos_cache[0:seq_len].unsqueeze(0).expand(1, -1, -1)
        sin_tg_slice = model_tg.model.sin_cache[0:seq_len].unsqueeze(0).expand(1, -1, -1)
        if not compare_tensors(cos_tg_slice, cos_pt, "RoPE Cos"): all_checks_passed = False
        if not compare_tensors(sin_tg_slice, sin_pt, "RoPE Sin"): all_checks_passed = False
    
    if all_checks_passed:
        num_layers_to_compare = min(len(hidden_states_tg) -1, len(hidden_states_pt) -1)
        for i in range(num_layers_to_compare):
            if not compare_tensors(hidden_states_tg[i + 1], hidden_states_pt[i + 1], f"Layer {i} Output"):
                print(f"\n‼️ DIVERGENCE DETECTED AT LAYER {i} ‼️")
                all_checks_passed = False
                break

    # C. Compare final logits
    if all_checks_passed:
        if not compare_tensors(logits_tg, logits_pt, "Final Logits"): 
            print(f"\n‼️ DIVERGENCE DETECTED AT FINAL LOGITS ‼️")
            all_checks_passed = False

    if all_checks_passed:
        print("\n🎉🎉🎉 All checks passed! The models match perfectly. 🎉🎉🎉")
    else:
        print("\n💥💥💥 Mismatch found! Please review the logs above. 💥💥💥")
        exit(1)