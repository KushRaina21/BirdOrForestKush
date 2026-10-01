import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATHS = [
    Path(__file__).resolve().parent / "model.pkl",
    Path(__file__).resolve().parent / "bird_model.pkl",
]
EXPECTED_LABELS = {"bird", "forest"}


def is_cat(path):
    return path.name.startswith("cat")


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
            labels = {str(label).lower() for label in model.dls.vocab}
            if not EXPECTED_LABELS.issubset(labels):
                continue
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
    green_pixels = float(
        ((g > r * 1.05) & (g > b * 1.02) & (g > 45)).mean()
    )

    return {
        "avg_r": avg_r,
        "avg_g": avg_g,
        "avg_b": avg_b,
        "mean_brightness": mean_brightness,
        "green_ratio": green_ratio,
        "warm_ratio": warm_ratio,
        "green_pixels": green_pixels,
    }


def fallback_classify(image: Image.Image):
    features = extract_features(image)
    vegetation_signal = min(features["green_pixels"] / 0.30, 1.0)
    green_dominance = min(max((features["green_ratio"] - 0.30) / 0.15, 0.0), 1.0)
    forest_score = vegetation_signal * 0.70 + green_dominance * 0.30
    bird_score = 1.0 - forest_score

    if forest_score >= bird_score:
        label = "Forest"
        confidence = min(max(0.5 + abs(forest_score - 0.5) * 0.45, 0.0), 0.95)
    else:
        label = "Bird"
        confidence = min(max(0.5 + abs(bird_score - 0.5) * 0.45, 0.0), 0.95)
    return label, round(confidence * 100, 1), features


def classify_image(image: Image.Image):
    if MODEL is not None:
        try:
            prediction, _, probs = MODEL.predict(image)
            label = str(prediction)
            confidence = float(probs.max().item() * 100)
            return label, round(confidence, 1), extract_features(image), "trained fastai model"
        except Exception:
            pass

    label, confidence, features = fallback_classify(image)
    return label, confidence, features, "color-based demo fallback"


st.set_page_config(page_title="Bird or Forest", page_icon="🦜", layout="centered")
st.title("Bird or Forest")
st.caption("Upload an image and the app will guess whether it looks more like a bird or a forest scene.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    label, confidence, features, prediction_source = classify_image(image)

    st.success(f"Prediction: {label} ({confidence}%)")
    st.caption(f"Prediction source: {prediction_source}")

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
st.write("The app uses the trained fastai model when it loads successfully. Otherwise, it labels the result as a color-based demo fallback.")
