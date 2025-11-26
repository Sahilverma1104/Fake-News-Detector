# trained_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Paths
FAKE_PATH = os.path.join("models", "Fake.csv")
TRUE_PATH = os.path.join("models", "True.csv")
MERGED_CSV_PATH = os.path.join("models", "fake_real_merged.csv")  # processed training data
MODEL_PATH = os.path.join("models", "fake_news_model.pkl")


def load_and_prepare_kaggle_data(fake_path: str, true_path: str) -> pd.DataFrame:
    """
    Kaggle dataset:
      - Fake.csv  → fake news
      - True.csv  → real news
    Columns usually: title, text, subject, date
    """
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake.csv not found at {fake_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True.csv not found at {true_path}")

    print("🔹 Loading Fake.csv ...")
    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = "FAKE"

    print("🔹 Loading True.csv ...")
    true_df = pd.read_csv(true_path)
    true_df["label"] = "REAL"

    # Combine
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Keep only useful columns (headline + label)
    # Tum chaho to 'text' bhi use kar sakte ho, but abhi sirf headline ke liye:
    required_cols = ["title", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

    df = df[required_cols].copy()
    df = df.dropna(subset=["title", "label"])

    return df


def preprocess_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map labels to numeric:
      FAKE -> 0
      REAL -> 1
    """
    mapping = {"FAKE": 0, "REAL": 1, "fake": 0, "real": 1}
    df["label_num"] = df["label"].map(mapping)

    if df["label_num"].isna().any():
        raise ValueError(
            "Some labels could not be mapped to 0/1. "
            "Make sure labels are 'FAKE' or 'REAL'."
        )

    return df


def train_and_save_model():
    print("🔹 Loading and preparing Kaggle data...")
    df = load_and_prepare_kaggle_data(FAKE_PATH, TRUE_PATH)
    df = preprocess_labels(df)

    # Save processed training data as a single CSV in models/ (as you asked)
    os.makedirs(os.path.dirname(MERGED_CSV_PATH), exist_ok=True)
    df.to_csv(MERGED_CSV_PATH, index=False)
    print(f"✅ Merged training data saved at {MERGED_CSV_PATH}")

    X = df["title"]          # only headline based
    y = df["label_num"]

    print("🔹 Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔹 Building pipeline (TF-IDF + Logistic Regression)...")
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                stop_words="english"
            )),
            ("clf", LogisticRegression(max_iter=300))
        ]
    )

    print("🔹 Training model on headlines...")
    pipeline.fit(X_train, y_train)

    print("🔹 Evaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
