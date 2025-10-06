# debug_comparison.py
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinygrad import Tensor, dtypes

from model.lfm2_modeling import LFM2ForCausalLM, LFM2Config, load_from_hf, hf_hub_download

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

    is_close = np.allclose(tg_np, pt_np, atol=1e-4, rtol=1e-3)
    if is_close:
        print("  ✅ MATCH: Tensors are numerically close.")
    else:
        print("  ❌ MISMATCH: Tensors are NOT close.")
    print("-" * (len(name) + 22))
    return is_close

# --- Main Comparison Logic ---
if __name__ == "__main__":
    REPO_ID = "LiquidAI/LFM2-350M"
    PROMPT = "The secret is"

    # 1. Load Hugging Face reference model
    print("Loading HF model...")
    model_hf = AutoModelForCausalLM.from_pretrained(REPO_ID, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    # 2. Load tinygrad model
    print("\nLoading tinygrad model...")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config_tg = LFM2Config.from_hf_config(config_dict)
    model_tg = LFM2ForCausalLM(config_tg)
    load_from_hf(model_tg, REPO_ID)

    # 3. Prepare identical inputs
    input_ids_pt = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    input_ids_tg = Tensor(input_ids_pt.numpy().astype(np.int32))
    seq_len = input_ids_pt.shape[1]

    # 4. Run full forward pass on both models
    print("\n\n--- Starting Step-by-Step tinygrad Comparison ---")
    
    # A. Run Hugging Face model to get all intermediate states
    with torch.no_grad():
        pt_out = model_hf(input_ids_pt, output_hidden_states=True)
        logits_pt = pt_out.logits
        # pt_out.hidden_states is a tuple: (initial_embeds, layer_0_out, ...)
        hidden_states_pt = pt_out.hidden_states
        
        # Get RoPE for comparison
        position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        cos_pt, sin_pt = model_hf.model.pos_emb(hidden_states_pt[0], position_ids)

    # B. Run tinygrad model
    tg_out = model_tg(input_ids_tg, output_hidden_states=True)
    logits_tg = tg_out.logits
    hidden_states_tg = tg_out.hidden_states

    # C. Compare initial embeddings
    if not compare_tensors(hidden_states_tg[0], hidden_states_pt[0], "Initial Embeddings"): exit()

    # D. Compare RoPE
    cos_tg_slice = model_tg.model.cos_cache[0:seq_len].unsqueeze(0).expand(1, -1, -1)
    sin_tg_slice = model_tg.model.sin_cache[0:seq_len].unsqueeze(0).expand(1, -1, -1)
    if not compare_tensors(cos_tg_slice, cos_pt, "RoPE Cos"): exit()
    if not compare_tensors(sin_tg_slice, sin_pt, "RoPE Sin"): exit()

    # E. Compare Layer Outputs
    for i in range(config_tg.num_hidden_layers):
        # HF hidden_states[0] is embedding, hidden_states[i+1] is output of layer i
        if not compare_tensors(hidden_states_tg[i + 1], hidden_states_pt[i + 1], f"Layer {i} Output"):
            print(f"\n‼️ DIVERGENCE DETECTED AT LAYER {i} ‼️")
            exit()

    # F. Compare final logits
    if not compare_tensors(logits_tg, logits_pt, "Final Logits"): 
        print(f"\n‼️ DIVERGENCE DETECTED AT FINAL LOGITS ‼️")
        exit()    

    print("\n🎉🎉🎉 All checks passed! The models match perfectly. 🎉🎉🎉")