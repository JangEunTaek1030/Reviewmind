"""
Compare simple text classification models for ReviewMind.

This script loads preprocessed review data, splits it into training and test
sets, trains three beginner-friendly baseline models, and prints a side-by-side
comparison of their evaluation metrics.

Expected input file:
    data/processed/processed_reviews.csv

Expected columns in the CSV:
    - cleaned_review
    - sentiment

Output file:
    outputs/figures/model_comparison_results.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def load_processed_data(csv_path: Path) -> pd.DataFrame:
    """Load the processed dataset and validate required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = ["cleaned_review", "sentiment"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in CSV: {column}")

    # Keep only rows where both text and label exist.
    df = df.dropna(subset=required_columns).copy()
    df["cleaned_review"] = df["cleaned_review"].astype(str)

    return df


def build_model_pipelines() -> dict[str, Pipeline]:
    """Create one TF-IDF + classifier pipeline per model."""
    return {
        "Multinomial Naive Bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
                ("classifier", MultinomialNB()),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
                ("classifier", LinearSVC()),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_train, X_test, y_train, y_test) -> dict[str, float]:
    """Train one model and return standard weighted classification metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def main() -> None:
    """Run end-to-end model comparison and save results."""
    input_path = Path("data/processed/processed_reviews.csv")
    output_path = Path("outputs/figures/model_comparison_results.csv")

    # 1) Load data
    df = load_processed_data(input_path)

    # 2) Split features and labels
    X = df["cleaned_review"]
    y = df["sentiment"]

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4) Build and compare models
    pipelines = build_model_pipelines()
    results = []

    for model_name, pipeline in pipelines.items():
        metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        results.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

    results_df = pd.DataFrame(results)

    # 5) Print clean comparison table
    print("\nModel Comparison Results")
    print("-" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 6) Save comparison table to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved comparison results to: {output_path}")


if __name__ == "__main__":
    main()
