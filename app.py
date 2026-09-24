import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATHS = [
    Path(__file__).resolve().parent / "model.pkl",
    Path(__file__).resolve().parent / "bird_model.pkl",
]


def load_fastai_model(candidate_paths):
    try:
        from fastai.vision.all import load_learner
    except Exception:
        return None

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            model = load_learner(path)
            return model
        except Exception:
            continue
    return None


MODEL = load_fastai_model(MODEL_PATHS)


def extract_features(image: Image.Image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    avg_r = float(r.mean())
    avg_g = float(g.mean())
    avg_b = float(b.mean())
    mean_brightness = float((r + g + b).mean() / 3.0)
    green_ratio = float(avg_g / max(avg_r + avg_g + avg_b, 1e-6))
    warm_ratio = float((avg_r + avg_b) / max(avg_g + avg_r + avg_b, 1e-6))

    return {
        "avg_r": avg_r,
        "avg_g": avg_g,
        "avg_b": avg_b,
        "mean_brightness": mean_brightness,
        "green_ratio": green_ratio,
        "warm_ratio": warm_ratio,
    }


def fallback_classify(image: Image.Image):
    features = extract_features(image)
    forest_score = features["green_ratio"] * 1.25 + (1.0 - features["warm_ratio"]) * 0.7
    bird_score = features["warm_ratio"] * 0.9 + (1.0 - features["green_ratio"]) * 0.8

    if forest_score >= bird_score:
        label = "Forest"
        confidence = min(max(forest_score, 0.0), 1.0)
    else:
        label = "Bird"
        confidence = min(max(bird_score, 0.0), 1.0)
    return label, round(confidence * 100, 1), features


def classify_image(image: Image.Image):
    if MODEL is not None:
        try:
            prediction, _, probs = MODEL.predict(image)
            label = str(prediction)
            confidence = float(probs.max().item() * 100)
            return label, round(confidence, 1), extract_features(image)
        except Exception:
            pass

    return fallback_classify(image)


st.set_page_config(page_title="Bird or Forest", page_icon="🦜", layout="centered")
st.title("Bird or Forest")
st.caption("Upload an image and the app will guess whether it looks more like a bird or a forest scene.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    label, confidence, features = classify_image(image)

    st.success(f"Prediction: {label} ({confidence}%)")

    with st.expander("Feature summary"):
        st.write({
            "Average red": round(features["avg_r"], 2),
            "Average green": round(features["avg_g"], 2),
            "Average blue": round(features["avg_b"], 2),
            "Brightness": round(features["mean_brightness"], 2),
            "Green ratio": round(features["green_ratio"], 3),
            "Warm color ratio": round(features["warm_ratio"], 3),
        })
else:
    st.info("Please upload an image to get a prediction.")

st.markdown("---")
st.write("This demo prefers a trained fastai model when available and falls back to a color-based heuristic if the saved model cannot be loaded.")
