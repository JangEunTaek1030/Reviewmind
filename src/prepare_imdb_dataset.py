"""Prepare a realistic IMDB sentiment dataset for ReviewMind.

This script downloads a small beginner-friendly subset of the
`stanfordnlp/imdb` dataset from Hugging Face, preprocesses the text,
and saves the final CSV file used by `src/train_ml.py`.
"""

from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset

from data_preprocessing import preprocess_dataframe


def map_label_to_sentiment(label: int) -> str:
    """Convert numeric IMDB labels into readable sentiment labels."""
    label_map = {0: "negative", 1: "positive"}
    if label not in label_map:
        raise ValueError(f"Unexpected label value: {label}")
    return label_map[label]


def main() -> None:
    """Download, preprocess, and save the IMDB subset for training."""
    output_path = Path("data/processed/processed_reviews.csv")

    # 1) Load the IMDB dataset from Hugging Face.
    imdb_dataset = load_dataset("stanfordnlp/imdb")

    # 2) Keep a small subset so beginners can run quickly.
    train_subset = imdb_dataset["train"].select(range(2000))
    test_subset = imdb_dataset["test"].select(range(500))

    # 3) Combine train and test subsets, then convert to pandas.
    combined_subset = concatenate_datasets([train_subset, test_subset])
    df = combined_subset.to_pandas()

    # 4) Convert label IDs (0/1) to readable labels.
    df["sentiment"] = df["label"].apply(map_label_to_sentiment)

    # 5) Rename text column to 'review' and keep needed columns.
    df = df.rename(columns={"text": "review"})
    df = df[["review", "sentiment"]]

    # 6) Apply existing preprocessing to create 'cleaned_review'.
    processed_df = preprocess_dataframe(df, text_column="review", label_column="sentiment")

    # 7) Save to the path expected by src/train_ml.py.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    # 8) Print helpful dataset diagnostics.
    print("Dataset preparation completed successfully.")
    print(f"Saved processed dataset to: {output_path}")
    print(f"Final dataset size: {len(processed_df)}")

    print("\nLabel distribution:")
    label_counts = processed_df["sentiment"].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = (count / len(processed_df)) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
