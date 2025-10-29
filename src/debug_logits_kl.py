# /src/debug_logits_kl.py

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# tinygrad imports
from tinygrad import Tensor, dtypes

# Use the new unified model loading
from model import MODEL_MAP

# --- Helper Functions ---
def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def calculate_kl_divergence(p_probs: np.ndarray, q_probs: np.ndarray) -> float:
    """Calculates the Kullback-Leibler divergence between two probability distributions."""
    # Add a small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    p_safe = np.clip(p_probs, epsilon, 1.0)
    q_safe = np.clip(q_probs, epsilon, 1.0)
    
    # The sum of P * log(P/Q)
    return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))

# --- Main Comparison Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug model logits via KL divergence after generation.")
    parser.add_argument("--model", type=str, default="LFM2", choices=MODEL_MAP.keys(), help="Model to debug.")
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-350M", help="Hugging Face model ID.")
    args = parser.parse_args()

    PROMPT = "The secret to life is"
    MAX_NEW_TOKENS = 50

    # 1. Load Hugging Face reference model
    print(f"Loading HF model: {args.model_id}...")
    model_hf = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load tinygrad model
    print(f"\nLoading tinygrad model: {args.model} from {args.model_id}...")
    CausalLM = MODEL_MAP[args.model]
    model_tg = CausalLM.from_pretrained(args.model_id)

    # 3. Prepare inputs
    input_ids_pt = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    input_ids_tg = Tensor(input_ids_pt.numpy().astype(np.int32))
    prompt_len = input_ids_pt.shape[1]

    # 4. Generate full sequences deterministically
    print("\n--- Generating full sequences (temperature=0) ---")
    print("Generating with Hugging Face model...")
    with torch.no_grad():
        output_hf = model_hf.generate(
            input_ids_pt, 
            max_new_tokens=MAX_NEW_TOKENS, 
            temperature=0.0, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        print(tokenizer.decode(output_hf[0], skip_special_tokens=False))

    print("Generating with tinygrad model...")
    output_tg_tensor = model_tg.generate(
        input_ids_tg, 
        max_new_tokens=MAX_NEW_TOKENS, 
        temperature=0.0, 
        do_sample=False
    )
    output_tg = output_tg_tensor.numpy()

    # 5. Compare generated sequences as a preliminary check
    print("\n--- Comparing generated token sequences ---")
    if not np.array_equal(output_hf, output_tg):
        print("❌ MISMATCH: Generated token sequences are different. Aborting KL check.")
        print(f"  HF ({output_hf.shape}): {tokenizer.decode(output_hf[0])}")
        print(f"  TG ({output_tg.shape}): {tokenizer.decode(output_tg[0])}")
        exit(1)
    else:
        print("✅ MATCH: Generated token sequences are identical.")
        print(f"Generated text: {tokenizer.decode(output_hf[0])}")

    # 6. Get logits for the entire generated sequence from both models
    print("\nCalculating full logits for both models...")
    with torch.no_grad():
        logits_pt_full = model_hf(output_hf).logits.squeeze(0).cpu().numpy()
    
    logits_tg_full = model_tg(Tensor(output_tg, dtype=dtypes.int32)).logits.squeeze(0).numpy()
    
    # 7. Perform KL divergence check for each generated token's logits
    print("\n--- Starting KL Divergence Check ---")
    total_kl = 0.0
    diverged = False
    # We check the logits that were used to *predict* each token in the generated sequence.
    # The logits at index `i` predict the token at `i+1`.
    # We start from the last token of the prompt (`prompt_len - 1`).
    for i in range(output_hf.shape[1] - 1):
        logits_pt = logits_pt_full[i, :]
        logits_tg = logits_tg_full[i, :]
        
        probs_pt = softmax(logits_pt)
        probs_tg = softmax(logits_tg)
        
        kl_div = calculate_kl_divergence(probs_pt, probs_tg)
        total_kl += kl_div
        
        token_id = output_hf[0, i + 1].item()
        token_str = tokenizer.decode([token_id])

        print(f"  Logits for Token[{i+1}] (ID: {token_id}, Text: '{token_str.strip()}'): KL(HF || TG) = {kl_div:.8f}")

        # A small KL divergence is expected due to floating point differences.
        # A large one indicates a numerical issue.
        if kl_div > 1e-3: 
            diverged = True

    avg_kl = total_kl / (output_hf.shape[1] - prompt_len)
    print(f"\nAverage KL Divergence over {MAX_NEW_TOKENS} tokens: {avg_kl:.8f}")
    
    if diverged:
        print("\n💥💥💥 High KL divergence detected! The models' output distributions differ significantly. 💥💥💥")
        exit(1)
    else:
        print("\n🎉🎉🎉 All checks passed! Logits are numerically very close across the generation. 🎉🎉🎉")