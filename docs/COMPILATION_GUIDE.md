# Guide: Optimizing Layers with TinyJit for Inference

This guide explains the compilation strategy used in OpenFormer to accelerate autoregressive decoding (inference) with `tinygrad`'s Just-In-Time (JIT) compiler. Understanding this is key for optimizing new or existing model layers.

## The Challenge: Dynamic Shapes in Inference

The core performance challenge for `tinygrad` in LLM inference is the dynamic shape of the input.
-   The initial **prefill** stage processes a prompt of variable length (e.g., 50 tokens).
-   The subsequent **decoding** stage generates one token at a time, with an input shape of `(batch_size, 1, hidden_size)`.

Because the sequence length changes at each step, a naive forward pass would cause the JIT compiler to re-compile the computation graph for every single token, leading to significant overhead and slow generation.

## The Solution: The Prefill/Decode Split

To overcome this, OpenFormer uses a **"Prefill/Decode Split"** strategy. We treat the two phases differently and apply `TinyJit` selectively to the parts of the model that operate on fixed-size inputs during the decoding phase.

-   **Prefill Phase (`seq_len > 1`):** We run a standard, non-compiled forward pass. This is only done once per generation, so the overhead is acceptable.
-   **Decoding Phase (`seq_len == 1`):** We call a separate, JIT-compiled forward pass. Since the input shape is always `(1, 1, hidden_dim)` and we manage state carefully, the JIT can create a highly optimized kernel that is reused for every subsequent token.

This compilation only affects autoregressive decoding. During training (`train_sft.py`), inputs are padded to a fixed `max_length`, so the entire `train_step` function can be safely compiled.

## Implementation Workflow

When optimizing a layer, you must first analyze how its state changes during decoding.

### 1. Analyze the Layer's State

For a single decoding step, does the layer's computation depend on a fixed-size input, or does it depend on a cache that grows with the sequence length?

-   **Fixed-Size State (O(1) Complexity):** The computation depends only on the current token's hidden state and a fixed-size cache (or no cache at all). Examples include an MLP block or the convolution operator in `LFM2ConvOperator`. These are the easiest to compile.
-   **Variable-Size State (O(N) Complexity):** The computation depends on a cache that grows with the sequence length, like the Key-Value (KV) cache in attention. These require a special pattern to compile correctly.

### 2. Path A: Compiling Fixed-Size Layers (The Easy Path)

If a layer's decoding logic is fixed-size, you can simply wrap its forward pass in `@TinyJit`.

A great example is the feed-forward network (FFN) portion of a decoder layer. In `lfm2_modeling.py`, the FFN logic is separated and JIT-compiled.

```python
# From src/model/lfm2_modeling.py

class LFM2DecoderLayer:
    def __init__(self, ...):
        # ... other initializations
        self._ffn_jit = TinyJit(self._ffn)

    def _ffn(self, hidden_states: Tensor):
        # This part is always fixed-size during decoding
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

    def __call__(self, hidden_states: Tensor, ...):
        # ... operator logic (attention or conv)
        
        # Dispatch to the correct FFN implementation
        if hidden_states.shape[1] > 1: # Prefill
            hidden_states = self._ffn(hidden_states)
        else: # Decode
            hidden_states = self._ffn_jit(hidden_states)
        
        return hidden_states, new_state
```

### 3. Path B: Compiling Variable-Size Layers (The Advanced Path)

Layers like attention, which use a growing KV cache, are more complex. To make them JIT-compatible, we use `UOp` symbolic variables. This tells `tinygrad` that a specific value (the current sequence length) will change at runtime, allowing it to create a generalized kernel.

Follow the pattern in `src/model/base_modeling.py`'s `BaseAttention`:

1.  **Separate Prefill and Decode Logic:** Create two distinct methods: `_forward_prefill` for variable-length inputs and `_forward_decoding` for single-token inputs.

2.  **Make the Cache an Internal State:** The growing cache (e.g., `cache_kv`) must be a member of the class (`self.cache_kv`) rather than being passed in and returned. This is essential for the JIT to track its state.

3.  **Use a `UOp` Symbolic Variable:** The `start_pos` (current sequence length) is the dynamic variable. We define it as a `UOp.variable` in the main `generate` loop and bind its value at each step.

4.  **JIT the Decoding Method Only:** Apply `@TinyJit` exclusively to the `_forward_decoding` method.

5.  **Dispatch in `__call__`:** The main `__call__` method checks the input sequence length (`q_len`) and calls the appropriate prefill or decoding method.

```python
# Simplified example from src/model/base_modeling.py

class BaseAttention:
    def __init__(self, ...):
        # ...
        self._forward_decoding_jit = TinyJit(self._forward_decoding)

    def _forward_decoding(self, hidden_states: Tensor, start_pos: UOp, ...):
        # Assumes hidden_states is (B, 1, D)
        # ... calculate q, k, v for the single token
        
        # Update the internal cache at the symbolic start_pos
        self.cache_kv[:, :, :, start_pos:start_pos+1, :].assign(...)
        
        # Read the *entire* cache up to the new length
        key_states = self.cache_kv[0, :, :, 0:start_pos+1, :]
        value_states = self.cache_kv[1, :, :, 0:start_pos+1, :]

        # Perform attention
        attn_output = Tensor.scaled_dot_product_attention(...)
        return self.o_proj(attn_output)

    def _forward_prefill(self, hidden_states: Tensor, ...):
        # Standard forward pass for a full sequence
        # ...
        # Initialize and fill the internal cache
        self.cache_kv = Tensor.zeros(...)
        self.cache_kv[:, :, :, 0:q_len, :].assign(...)
        # ...
        return self.o_proj(attn_output), present_kv
    
    def __call__(self, hidden_states: Tensor, ..., start_pos: int | UOp):
        _, q_len, _ = hidden_states.shape

        if q_len > 1: # Prefill
            return self._forward_prefill(hidden_states, ...)
        else: # Decode
            output = self._forward_decoding_jit(hidden_states.contiguous(), start_pos, ...)
            return output, None
```

## Important Caveats

-   **Compile Layers, Not the Whole Model:** In our testing, attempting to JIT the entire model's forward pass at once can lead to numerical correctness bugs. The most stable and effective approach is to compile individual layers or sub-blocks independently.
-   **Ensure Tensors are `contiguous`:** Before passing a tensor to a JIT-compiled function, it's often necessary to call `.contiguous()` on it to ensure its memory layout is compatible with the optimized kernel.
