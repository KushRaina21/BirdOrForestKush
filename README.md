---
title: TestingFirstProject
emoji: 👀
colorFrom: red
colorTo: gray
sdk: static
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## CodeLlama inference helper

Files added in this workspace:
- `requirements.txt` — Python packages to install
- `run_inference.py` — small script to load a CodeLlama model and generate output

Usage (Kaggle notebook or local):

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) (Optional) Set a model name. Default is `codellama/CodeLlama-7b-hf`.

```bash
export MODEL_NAME=codellama/CodeLlama-7b-hf
```

3) Run the script:

```bash
python run_inference.py
```

Notes:
- On Kaggle, prefer the GPU runtime and install packages in a notebook cell.
- If GPU memory is insufficient, use a smaller CodeLlama variant or enable 8-bit loading via `bitsandbytes` and `transformers` quantization features.
- Model weights are pulled from Hugging Face; ensure you have access if a model requires an agreement.

