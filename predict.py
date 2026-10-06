"""Reusable prediction helpers."""
import os
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_pipeline.pkl"
KNOWN_EMOTIONS = {"happy", "sad", "angry", "neutral"}


def load_model(path=MODEL_PATH):
    """Load the saved sklearn pipeline. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def predict_emotion(text, model=None):
    """Return (emotion, confidence_percent) for a piece of text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty text")
    model = model or load_model()
    probs = model.predict_proba([text])[0]
    idx = int(probs.argmax())
    emotion = str(model.classes_[idx])
    if emotion not in KNOWN_EMOTIONS:
        raise ValueError(f"Unknown emotion class: {emotion}")
    return emotion, float(probs[idx]) * 100


def all_probabilities(text, model):
    """Return {emotion: probability_percent} for every class."""
    probs = model.predict_proba([text])[0]
    return {str(c): float(p) * 100 for c, p in zip(model.classes_, probs)}
