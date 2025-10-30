# Guide: Confirming Model and Hardware Support

OpenFormer is designed to run on a wide variety of hardware by leveraging tinygrad. While we test several common models and GPUs, the vast number of hardware combinations (AMD, Intel, Apple Silicon) and model variants (e.g., Qwen2-7B vs. Qwen2-0.5B) makes it impossible for us to test everything.

This guide empowers you to validate OpenFormer on your specific hardware and for untested model variants from a supported family. By following these steps, you can confirm compatibility and help the community by reporting your findings.

## Step 1: The Tinygrad Sanity Check

Before testing OpenFormer, you must confirm that tinygrad itself is running correctly on your machine. This simple test isolates issues with your drivers or tinygrad installation from any potential bugs in the OpenFormer library.

1.  Ensure you install all dependencies (see: `README.md` getting started).
2.  Save the following code as `sanity_check.py`:

    ```python
    # sanity_check.py
    from tinygrad import Tensor, Device
    import numpy as np

    print(f"--- Running tinygrad sanity check on device: {Device.DEFAULT} ---")

    try:
        # Create two random tensors
        a = Tensor.rand(128, 128)
        b = Tensor.rand(128, 128)

        # Perform a matrix multiplication and realize the result
        c = (a @ b).realize()

        # Move the result to the CPU and print a value to confirm
        c_numpy = c.numpy()
        print(f"✅ Sanity check PASSED!")
        print(f"   Matrix multiplication successful.")
        print(f"   Value at [0, 0]: {c_numpy[0, 0]}")

    except Exception as e:
        print(f"❌ Sanity check FAILED!")
        print(f"   An error occurred during a basic tinygrad operation.")
        print(f"   Please check your tinygrad installation, GPU drivers, and device setup.")
        print(f"   Error: {e}")

    ```
3.  Run the script from your terminal:

    ```bash
    python sanity_check.py
    ```

If the script prints `✅ Sanity check PASSED!`, your tinygrad installation is working correctly. You can proceed to the next step. If it fails, the issue lies with your environment, not OpenFormer. Please consult the [tinygrad repository](https://github.com/tinygrad/tinygrad) for troubleshooting.

## Step 2: Validating an OpenFormer Model

Once tinygrad is confirmed to be working, you can test any model from our supported architecture list. The most rigorous way to do this is by using our numerical verification scripts.

We recommend starting with `debug_prefilling.py` as it's the simplest test that runs a full forward pass.

1.  **Choose a model to test.** For example, let's say you want to test the untested `Qwen/Qwen2-7B-Instruct` model on your AMD GPU.
2.  **Run the appropriate debug script.** The `--model` argument should match the key in `src/model/__init__.py`, and `--model_id` should be the Hugging Face repository.

    ```bash
    # Test the prefilling (single forward pass) stage
    python src/debug_prefilling.py --model Qwen2 --model_id "Qwen/Qwen2-7B-Instruct"

    # If prefilling succeeds, test the decoding (generation) stage
    python src/debug_decoding.py --model Qwen2 --model_id "Qwen/Qwen2-7B-Instruct"

    # For the most robust check, test the KL divergence of logits
    python src/debug_logits_kl.py --model Qwen2 --model_id "Qwen/Qwen2-7B-Instruct"
    ```

A successful run will print `✅ MATCH` for all tensor comparisons and end with a `🎉🎉🎉 All checks passed!` message. If all debug scripts pass, your hardware and the specific model variant are fully compatible!

You can also perform simpler checks:
-   **Inference:** Run `src/run.py` to see if text generation works without crashing.
-   **Training:** Run `src/train_sft.py` with a small `--max_steps` value to validate the training loop.

## Step 3: Found a Bug? How to Report It

If the tinygrad sanity check passes but one of the OpenFormer scripts fails, you've likely found a compatibility bug! Please open an issue on our GitHub repository with the following structured format to help us resolve it quickly.

---

### Bug Report Template

**1. Model Information**
*   **Model Family (`--model`):** [e.g., Qwen2, Llama3, Granite4]
*   **Model ID (`--model_id`):** [e.g., `Qwen/Qwen2-7B-Instruct`]

**2. Hardware & Environment**
*   **GPU:** [e.g., AMD RX 6700 XT, Intel Arc A770, Apple M2 Pro]
*   **CPU:** [e.g., AMD Ryzen 7 5800X, Intel Core i9-13900K]
*   **OS:** [e.g., Ubuntu 22.04, Windows 11, macOS Sonoma]
*   **tinygrad version:** [Run `pip show tinygrad` to find this]

**3. Test Script Used**
*   **Script Name:** [e.g., `debug_prefilling.py`, `run.py`, `train_sft.py`]

**4. Command Used**
*Please provide the exact command you ran.*
```bash
# Paste the full command here
python src/debug_prefilling.py --model Qwen2 --model_id "Qwen/Qwen2-7B-Instruct"
```

**5. Error Log / Output**
*Please paste the complete, unedited output from your terminal below.*
<details>
<summary>Click to expand error log</summary>

```
<-- Paste the full terminal output here -->
```

</details>

---

Thank you for helping us expand the reach of open-source LLMs to more hardware and models