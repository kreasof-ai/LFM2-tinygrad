# debug_comparison.py
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinygrad import Tensor, dtypes

from lfm2_modelling import LFM2ForCausalLM, LFM2Config, load_from_hf, hf_hub_download

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
    # Get config from HF model
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

    # --- 4. Get all intermediate outputs from HF model ---
    hf_outputs = {}
    with torch.no_grad():
        # A. Embeddings
        hf_outputs["embeds"] = model_hf.model.embed_tokens(input_ids_pt)

        # B. Rotary Embeddings
        position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        cos_hf, sin_hf = model_hf.model.pos_emb(hf_outputs["embeds"], position_ids)
        hf_outputs["cos"] = cos_hf
        hf_outputs["sin"] = sin_hf

        # C. Full forward pass to get hidden states
        # The first element of `hidden_states` is the initial embedding
        # `hidden_states[i+1]` is the output of `layer i`
        pt_forward_out = model_hf.model(input_ids_pt, output_hidden_states=True)
        print(len(pt_forward_out.hidden_states))
        for i in range(config_tg.num_hidden_layers):
            hf_outputs[f"layer_{i}_out"] = pt_forward_out.hidden_states[i + 1]

        # D. Final Norm and Logits
        hf_outputs["final_norm"] = model_hf.model.embedding_norm(pt_forward_out.hidden_states[-1])
        hf_outputs["logits"] = model_hf.lm_head(hf_outputs["final_norm"])

    # --- 5. Step-by-step forward pass for tinygrad model and compare ---
    print("\n\n--- Starting Step-by-Step tinygrad Comparison ---")

    # A. Compare Embeddings
    h_tg = model_tg.model.embed_tokens(input_ids_tg)
    if not compare_tensors(h_tg, hf_outputs["embeds"], "Initial Embeddings"): exit()

    # B. Compare Rotary Embeddings
    cos_tg, sin_tg = model_tg.model.rotary_emb(h_tg, seq_len, 0)
    if not compare_tensors(cos_tg, hf_outputs["cos"], "RoPE Cos"): exit()
    if not compare_tensors(sin_tg, hf_outputs["sin"], "RoPE Sin"): exit()

    # C. Compare Layer Outputs
    mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1).realize() if seq_len > 1 else None
    
    # Initialize dummy past states for prefill
    past_states = [None] * len(model_tg.model.layers)
    
    for i, layer in enumerate(model_tg.model.layers):
        h_tg, _ = layer(h_tg, mask, past_states[i], (cos_tg, sin_tg))
        if i + 1 == len(model_tg.model.layers):
            h_tg = model_tg.model.norm(h_tg)
        if not compare_tensors(h_tg, hf_outputs[f"layer_{i}_out"], f"Layer {i} Output"):
            print(f"\n‼️ DIVERGENCE DETECTED AT LAYER {i} ‼️")
            exit()

    # --- NEW STEP: Now compare the final normed output separately ---
    # The loop finished. h_tg is now the raw output of the final layer.
    final_h_tg_normed = model_tg.model.norm(h_tg)
    if not compare_tensors(final_h_tg_normed, hf_outputs["final_norm"], "Final Norm Output"):
        print(f"\n‼️ DIVERGENCE DETECTED AT FINAL NORM ‼️")
        exit()

    # --- FINAL STEP: Calculate and compare logits using the correctly normed tensor ---
    logits_tg = model_tg.lm_head(final_h_tg_normed) # Use the correctly normed tensor
    if not compare_tensors(logits_tg, hf_outputs["logits"], "Final Logits"): 
        print(f"\n‼️ DIVERGENCE DETECTED AT FINAL LOGITS ‼️")
        exit()

    print("\n🎉🎉🎉 All checks passed! The models match perfectly. 🎉🎉🎉")