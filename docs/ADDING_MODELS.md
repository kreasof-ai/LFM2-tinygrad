# Guide: Adding and Testing New Models in OpenFormer

This guide provides a step-by-step walkthrough for contributing new transformer model architectures to the OpenFormer library. The core philosophy of OpenFormer is to maximize code reuse through a powerful and extensible `base_modeling.py`.

## Part 1: Architectural Analysis

Before writing any code, the first step is to analyze the new model's architecture to determine how it fits into the OpenFormer framework.

### The "Standard" Architecture

Most modern decoder-only transformers follow a similar "pre-norm" structure:

1.  Input goes through an `RMSNorm` layer.
2.  The normalized output is passed to a `Self-Attention` block.
3.  The output of the attention block is added to the original input (residual connection).
4.  This new tensor goes through another `RMSNorm` layer.
5.  The normalized output is passed to an `MLP` (Feed-Forward) block.
6.  The output of the MLP is added to the input from step 3 (second residual connection).

OpenFormer provides robust, reusable components for this pattern in `src/model/base_modeling.py`:
-   `BaseAttention`: A standardized attention module supporting Grouped-Query Attention (GQA), optional QK Normalization, and RoPE.
-   Note that Tinygrad `scaled_dot_product_attention` don't support softmax scale at this moment. So, you need to implement manual attention like `gemma3_modeling.py` or `falconh1_modeling.py` if you need custom softmax scale.
-   `BaseMLP`: A standardized SwiGLU MLP.
-   `BaseModel` & `BaseForCausalLM`: A powerful skeleton that handles weight loading, generation logic, quantization, and tying all the components together.

**Models that fit this pattern:** Llama 3, Qwen2, Qwen3, Hunyuan. These models can be implemented with very little new code, primarily by wiring up the base components correctly.

### The "Hybrid" or "Custom" Architecture

Some models introduce unique components or change the standard layer structure. These require custom implementations.

-   **Hybrid Layer Structure:** The model might mix different types of layers.
    -   **Example: `LFM2`** uses a mix of standard attention layers and custom `LFM2ConvOperator` convolution layers. Its `LFM2DecoderLayer` must conditionally choose which operator to use.
-   **Custom Components:** The model might use a non-standard normalization, activation, or attention mechanism.
    -   **Example: `Gemma3`** uses a custom `Gemma3RMSNorm` that scales with `1 + weight`, uses `gelu` activation in its MLP, and has a unique attention calculation with a `query_pre_attn_scalar`.
    -   **Example: `Granite4`** is a hybrid of Mamba and Attention layers, requiring a completely custom `Granite4MambaLayer`.

If a model is hybrid or custom, you must implement these unique components yourself. However, you can still reuse parts of `base_modeling.py` where possible.

## Part 2: Implementation Guide

### Path A: Adding a Standard Model (e.g., Llama-like)

This is the most common path. We will use `llama_modeling.py` as our reference.

**1. Create the new file:**
Create `src/model/newmodel_modeling.py`.

**2. Define the Config:**
Inherit from `BaseConfig` and implement the `from_hf_config` class method to map keys from the Hugging Face `config.json`.

```python
# src/model/newmodel_modeling.py

from model.base_modeling import BaseConfig, BaseAttention, BaseMLP, BaseModel, BaseForCausalLM
# ... other imports

@dataclass
class NewModelConfig(BaseConfig):
    # Add any model-specific config fields if needed
    pass

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "NewModelConfig":
        return cls(
            # Map Hugging Face config keys to our BaseConfig fields
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            # ... etc.
        )
```

**3. Define the Decoder Layer:**
Create a `NewModelDecoderLayer` that wires together `BaseAttention` and `BaseMLP` according to the model's pre-norm architecture.

```python
# src/model/newmodel_modeling.py

class NewModelDecoderLayer:
    def __init__(self, config: NewModelConfig, linear_class: Type):
        self.self_attn = BaseAttention(config, linear_class)
        self.mlp = BaseMLP(config, linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states, ...):
        residual = hidden_states
        normed_hidden = self.input_layernorm(hidden_states)
        attn_output, new_kv = self.self_attn(normed_hidden, ...)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        normed_hidden = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden)
        hidden_states = residual + mlp_output
        return hidden_states, new_kv
```

**4. Define the Model Wrapper:**
Inherit from `BaseModel` and tell it how to create a decoder layer.

```python
# src/model/newmodel_modeling.py

class NewModel(BaseModel):
    def _create_decoder_layer(self, config: BaseConfig, linear_class: Type, layer_idx: int):
        return NewModelDecoderLayer(config, linear_class)
```

**5. Define the CausalLM Head Model:**
Inherit from `BaseForCausalLM` and implement the three required methods. This class ties everything together.

```python
# src/model/newmodel_modeling.py

class NewModelForCausalLM(BaseForCausalLM):
    def _create_model(self, config: BaseConfig, linear_class: Type) -> BaseModel:
        return NewModel(config, linear_class)
    
    @classmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        # Logic to download and parse config.json
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        return NewModelConfig.from_hf_config(config_dict)
    
    def _get_key_map(self) -> dict:
        # This is CRITICAL. It maps HF weight names to OpenFormer weight names.
        # Format: {"hugging_face_name": "openformer_name"}
        key_map = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i in range(self.config.num_hidden_layers):
            p = f"model.layers.{i}"
            key_map.update({
                f"{p}.input_layernorm.weight": f"{p}.input_layernorm.weight",
                f"{p}.self_attn.q_proj.weight": f"{p}.self_attn.q_proj.weight",
                # ... etc for all weights in a layer
            })
        return key_map
```

**6. Register the Model:**
Finally, add your new model to the `MODEL_MAP` in `src/model/__init__.py`.

```python
# src/model/__init__.py

MODEL_MAP = {
    "LFM2": lfm2_modeling.LFM2ForCausalLM,
    # ... other models
    "NewModel": newmodel_modeling.NewModelForCausalLM, # Add your model here
}
```

### Path B: Adding a Hybrid/Custom Model

If your model is not standard, you will need to implement its unique components first, but the overall structure remains the same. See `lfm2_modeling.py` or `gemma3_modeling.py` for excellent examples of how to mix custom and base components.

## Part 3: Testing and Verification

Once implemented, you must verify that your model produces numerically identical outputs to the official Hugging Face implementation. We provide three debug scripts for this purpose.

### Test 1: Prefilling Stage (`debug_prefilling.py`)

This test checks the forward pass for a given prompt. It compares the initial embeddings, the output of each decoder layer, and the final logits. It's the best place to start debugging.

-   **What it does:** Verifies the correctness of a single, full forward pass.
-   **How to run:**
    ```bash
    python src/debug_prefilling.py --model NewModel --model_id "hf/new-model-id"
    ```
-   **Caveats:** This script assumes the Hugging Face model follows a standard structure (e.g., `model.model.layers`). For highly custom models, you might need to slightly modify the script to correctly access the HF model's intermediate outputs for comparison.

### Test 2: Decoding Stage (`debug_decoding.py`)

This test builds on the prefilling test. It first runs the prefill stage and then generates several tokens one by one, comparing the logits and the state of the KV cache at *each step*.

-   **What it does:** Verifies the correctness of the autoregressive generation loop and KV cache management.
-   **How to run:**
    ```bash
    python src/debug_decoding.py --model NewModel --model_id "hf/new-model-id"
    ```
-   **Caveats:** Like the prefilling test, this assumes the HF model's KV cache (`past_key_values`) has a standard format. If the test fails here but prefilling passes, the issue is likely in your KV cache handling.

### Test 3: Logits KL Divergence (`debug_logits_kl.py`)

This is the most robust and "unassuming" test. It does not inspect any intermediate states like layer outputs or the KV cache. Instead, it performs a full, deterministic generation with both the OpenFormer and Hugging Face models and then compares the final output distributions.

-   **What it does:** Generates a sequence of tokens and calculates the Kullback-Leibler (KL) divergence between the logit distributions produced by each model for each token. A very low KL divergence indicates the models are numerically equivalent.
-   **How to run:**
    ```bash
    python src/debug_logits_kl.py --model NewModel --model_id "hf/new-model-id"
    ```
-   **Why it's reliable:** This test only depends on the `generate()` and `forward()` methods. It will pass as long as the final logits are correct, regardless of how differently the models are implemented internally. It is the ultimate confirmation of correctness.

By following these implementation and testing steps, you can confidently add new, numerically verified models to the OpenFormer ecosystem.