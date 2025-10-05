# LFM2 (Liquid Foundation Model 2) tinygrad Implementation

This version is adapted to load pretrained weights from Hugging Face Hub. Not finished yet, there's significant output degradation.

## Get started

- Install dependencies:

    `pip install tinygrad torch transformers huggingface_hub safetensors tqdm`


## Acknowledgment

> Heavily inspired from https://github.com/kyegomez/LFM2 and official https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py implementation

## Disclaimer

- Empirical test with `compare.py` shows huggingface implementation keep double final norm behavior (which is weird). First inside final layer, then just before the lm_head.