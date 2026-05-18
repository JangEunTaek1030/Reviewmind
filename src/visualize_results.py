"""
Visualize model comparison results for ReviewMind.

This script creates two beginner-friendly visualizations:
1) A grouped bar chart for model metrics (accuracy, precision, recall, F1-score)
2) A confusion matrix for a TF-IDF + Logistic Regression classifier

Input files:
    - outputs/figures/model_comparison_results.csv
    - data/processed/processed_reviews.csv

Output files:
    - outputs/figures/model_comparison.png
    - outputs/figures/confusion_matrix.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_comparison_results(csv_path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Load model comparison metrics from CSV using Python's csv module."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Model comparison CSV not found: {csv_path}")

    model_names: list[str] = []
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = ["model", "accuracy", "precision", "recall", "f1_score"]
        if not reader.fieldnames:
            raise ValueError("Model comparison CSV is empty or missing headers.")

        for column in required_columns:
            if column not in reader.fieldnames:
                raise ValueError(f"Missing required column in comparison CSV: {column}")

        for row in reader:
            model_names.append(row["model"])
            metrics["accuracy"].append(float(row["accuracy"]))
            metrics["precision"].append(float(row["precision"]))
            metrics["recall"].append(float(row["recall"]))
            metrics["f1_score"].append(float(row["f1_score"]))

    if not model_names:
        raise ValueError("Model comparison CSV has no data rows.")

    return model_names, metrics


def plot_model_comparison(model_names: list[str], metrics: dict[str, list[float]], output_path: Path) -> None:
    """Create and save a grouped bar chart for model performance metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = ["accuracy", "precision", "recall", "f1_score"]
    x_positions = list(range(len(model_names)))
    bar_width = 0.2

    plt.figure(figsize=(11, 6))

    for index, metric_name in enumerate(metric_labels):
        offset_positions = [x + (index - 1.5) * bar_width for x in x_positions]
        plt.bar(offset_positions, metrics[metric_name], width=bar_width, label=metric_name)

    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(x_positions, model_names, rotation=15, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_processed_reviews(csv_path: Path) -> tuple[list[str], list[str]]:
    """Load cleaned review text and sentiment labels from the processed dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed reviews CSV not found: {csv_path}")

    texts: list[str] = []
    labels: list[str] = []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = ["cleaned_review", "sentiment"]

        if not reader.fieldnames:
            raise ValueError("Processed reviews CSV is empty or missing headers.")

        for column in required_columns:
            if column not in reader.fieldnames:
                raise ValueError(f"Missing required column in processed reviews CSV: {column}")

        for row in reader:
            review = row.get("cleaned_review", "")
            sentiment = row.get("sentiment", "")

            if review is None or sentiment is None:
                continue

            review_text = str(review).strip()
            sentiment_text = str(sentiment).strip()

            if review_text and sentiment_text:
                texts.append(review_text)
                labels.append(sentiment_text)

    if not texts:
        raise ValueError("Processed reviews CSV has no valid rows after filtering missing values.")

    return texts, labels


def build_logistic_regression_pipeline() -> Pipeline:
    """Build the same TF-IDF + Logistic Regression pipeline used in model comparison."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def create_confusion_matrix_figure(processed_data_path: Path, output_path: Path) -> None:
    """Train Logistic Regression and save a confusion matrix from the test split."""
    texts, labels = load_processed_reviews(processed_data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = build_logistic_regression_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ordered_labels = sorted(set(labels))
    cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordered_labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run all visualization steps and save output figures."""
    comparison_csv_path = Path("outputs/figures/model_comparison_results.csv")
    processed_reviews_path = Path("data/processed/processed_reviews.csv")

    model_comparison_png_path = Path("outputs/figures/model_comparison.png")
    confusion_matrix_png_path = Path("outputs/figures/confusion_matrix.png")

    model_names, metrics = load_comparison_results(comparison_csv_path)
    plot_model_comparison(model_names, metrics, model_comparison_png_path)

    create_confusion_matrix_figure(processed_reviews_path, confusion_matrix_png_path)

    print("Figures generated successfully:")
    print(f"- {model_comparison_png_path}")
    print(f"- {confusion_matrix_png_path}")


if __name__ == "__main__":
    main()
