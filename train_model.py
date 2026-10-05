"""Train the TF-IDF + Logistic Regression emotion classifier.

Usage:  python train_model.py
"""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import preprocess_text

DATA_PATH = os.path.join("data", "emotion_dataset.csv")
MODEL_PATH = os.path.join("models", "emotion_pipeline.pkl")
CM_PATH = os.path.join("results", "confusion_matrix.png")
REPORT_PATH = os.path.join("results", "evaluation_report.txt")
LABELS = ["happy", "sad", "angry", "neutral"]


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1-2. Load dataset and validate columns
    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found at {DATA_PATH}. Run: python generate_dataset.py")
    df = pd.read_csv(DATA_PATH)
    if not {"text", "emotion"}.issubset(df.columns):
        sys.exit("Dataset must contain the columns: text, emotion")

    # 3-4. Remove missing values and duplicates
    df = df[["text", "emotion"]].dropna()
    df["emotion"] = df["emotion"].str.strip().str.lower()
    df = df[df["emotion"].isin(LABELS)].drop_duplicates()

    # 5. Pre-process text; drop duplicates after cleaning to limit train/test leakage
    df["clean"] = df["text"].apply(preprocess_text)
    df = df[df["clean"].str.len() > 0].drop_duplicates(subset=["clean"])
    print(f"Samples after cleaning: {len(df)}")
    print(df["emotion"].value_counts().to_string(), "\n")

    # 6. Reproducible split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["emotion"], test_size=0.2, random_state=42, stratify=df["emotion"])

    # 7. Pipeline: raw text -> preprocessing -> TF-IDF -> Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    # 8-10. Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    summary = (
        "Voice-Based Emotion Detection - Evaluation Report\n"
        "(Trained on an EDUCATIONAL / DEMO synthetic dataset; scores are optimistic\n"
        " and do not represent performance on real-world speech.)\n\n"
        f"Train samples : {len(X_train)}\nTest samples  : {len(X_test)}\n\n"
        f"Accuracy            : {acc:.4f}\nPrecision (weighted): {p:.4f}\n"
        f"Recall (weighted)   : {r:.4f}\nF1-score (weighted) : {f1:.4f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows=true, cols=predicted, order={LABELS}):\n{cm}\n"
    )
    print(summary)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    # 12. Confusion matrix image
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Emotion Classifier")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()

    # 11. Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}\nConfusion matrix saved to {CM_PATH}")


if __name__ == "__main__":
    main()
