from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("Data/cleaned_data.csv")
MODEL_PATH = Path("models/spam_message_model.joblib")
TEXT_COLUMN = "clean_text"
FALLBACK_TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
RANDOM_STATE = 42


def find_best_threshold(y_true, spam_scores) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, spam_scores)
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold, p, r in zip(thresholds, precision[:-1], recall[:-1]):
        if p + r == 0:
            continue

        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def train() -> None:
    df = pd.read_csv(DATA_PATH)
    text_column = TEXT_COLUMN if TEXT_COLUMN in df.columns else FALLBACK_TEXT_COLUMN
    df = df[[text_column, LABEL_COLUMN]].dropna()
    df[text_column] = df[text_column].astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    spam_count = int(df[LABEL_COLUMN].sum())
    clean_count = int((df[LABEL_COLUMN] == 0).sum())
    print(f"Rows: {len(df):,}")
    print(f"Spam: {spam_count:,}")
    print(f"Not spam: {clean_count:,}")
    print(f"Spam ratio: {spam_count / len(df):.2%}")

    x_train, x_valid, y_train, y_valid = train_test_split(
        df[text_column],
        df[LABEL_COLUMN],
        test_size=0.2,
        stratify=df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=200_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    spam_scores = model.predict_proba(x_valid)[:, 1]
    threshold, best_f1 = find_best_threshold(y_valid, spam_scores)
    y_pred = (spam_scores >= threshold).astype(int)

    print(f"\nBest validation threshold: {threshold:.3f}")
    print(f"Best validation F1: {best_f1:.3f}")
    print(f"ROC AUC: {roc_auc_score(y_valid, spam_scores):.3f}")
    print("\nClassification report:")
    print(classification_report(y_valid, y_pred, target_names=["not_spam", "spam"]))

    artifact = {
        "model": model,
        "threshold": threshold,
        "label": LABEL_COLUMN,
        "text_column": text_column,
        "metrics": {
            "validation_f1": float(f1_score(y_valid, y_pred)),
            "validation_roc_auc": float(roc_auc_score(y_valid, spam_scores)),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
