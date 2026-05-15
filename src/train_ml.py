"""
Train a baseline sentiment classifier for ReviewMind.

This script loads preprocessed review data, trains a TF-IDF + Logistic Regression
model, evaluates it, and saves the trained pipeline for later use.

Expected input file:
    data/processed/processed_reviews.csv

Expected columns in the CSV:
    - cleaned_review
    - sentiment
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_processed_data(csv_path: Path) -> pd.DataFrame:
    """Load the processed review dataset from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = ["cleaned_review", "sentiment"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in CSV: {column}")

    df = df.dropna(subset=required_columns).copy()
    df["cleaned_review"] = df["cleaned_review"].astype(str)

    return df


def build_pipeline() -> Pipeline:
    """Create a simple TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def main() -> None:
    """Run model training, evaluation, and saving."""
    input_path = Path("data/processed/processed_reviews.csv")
    output_path = Path("outputs/models/ml_baseline_model.pkl")

    # 1) Load data
    df = load_processed_data(input_path)

    # 2) Split features and labels
    X = df["cleaned_review"]
    y = df["sentiment"]

    # 3) Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4) Train model
    model = build_pipeline()
    model.fit(X_train, y_train)

    # 5) Evaluate model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nModel Evaluation Results")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report")
    print("-" * 30)
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6) Beginner-friendly error analysis
    # Note: this dataset currently only has a "cleaned_review" column.
    # We use it as the review text to inspect model mistakes.
    error_analysis_df = pd.DataFrame(
        {
            "test_review_text": X_test.values,
            "true_sentiment": y_test.values,
            "predicted_sentiment": y_pred,
        }
    )
    error_analysis_df["is_correct"] = (
        error_analysis_df["true_sentiment"] == error_analysis_df["predicted_sentiment"]
    )

    print("\nError Analysis (first 10 test examples)")
    print("-" * 30)
    print(error_analysis_df.head(10).to_string(index=False))

    # 7) Save model pipeline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(model, f)

    print(f"Saved trained model to: {output_path}")


if __name__ == "__main__":
    main()
