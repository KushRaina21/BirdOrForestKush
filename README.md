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

This Streamlit app predicts whether an uploaded photo looks more like a bird or a forest scene.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Deploy this repository from Streamlit Community Cloud with `app.py` as the main file. The app requires the packages listed in `requirements.txt`.

## Notes

- The app checks `model.pkl` and then `bird_model.pkl` for a compatible Bird-or-Forest fastai learner.
- If no compatible learner loads, the UI identifies its result as a color-based demo fallback.
- The included serialized learners currently require additional compatibility work, so verify the prediction source in the live app before treating results as trained-model output.

