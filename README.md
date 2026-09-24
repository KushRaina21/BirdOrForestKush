---
title: BirdOrForestKush
emoji: 🐦
colorFrom: green
colorTo: blue
sdk: streamlit
pinned: false
short_description: Bird or forest image classifier demo
---

# BirdOrForestKush

This Hugging Face Space hosts a simple image classification demo that predicts whether an uploaded photo looks more like a bird or a forest scene.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app loads a demo model file if present and falls back to a lightweight heuristic when no trained model is available.
- You can replace `model.pkl` with a real trained classifier later.

