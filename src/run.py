import json
# Third-party imports
from huggingface_hub import hf_hub_download
from model.lfm2_modeling import LFM2Config, LFM2ForCausalLM, generate, load_from_hf
from transformers import AutoTokenizer

# tinygrad imports
from tinygrad import Tensor
from tinygrad.helpers import getenv
from tinygrad import Device

print("DEVICE:", Device.DEFAULT)

# For reproducible tests
if getenv("SEED"):
    Tensor.manual_seed(getenv("SEED"))

if __name__ == "__main__":
    REPO_ID = "LiquidAI/LFM2-350M"
    print(f"--- Loading LFM2 Model: {REPO_ID} ---")

    # 1. Download and load the configuration
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = LFM2Config.from_hf_config(config_dict)
    print("\nModel configuration:")

    for i in range(config.num_hidden_layers):
        is_attention = i in config.full_attn_idxs
        print(f"Layer {i}: {'Attention' if is_attention else 'Convolution'}")

    # 2. Initialize the model structure
    print("\nInitializing model architecture...")
    model = LFM2ForCausalLM(config)

    # 3. Load the pretrained weights
    load_from_hf(model, REPO_ID, filename="model.safetensors")

    # 4. Load the tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    # 5. Run text generation
    prompt = "The secret to a long and happy life is"
    generated_text = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=50,
        temperature=0.3
    )