"""Prepare a realistic IMDB sentiment dataset for ReviewMind.

This script downloads a small beginner-friendly subset of the
`stanfordnlp/imdb` dataset from Hugging Face, preprocesses the text,
and saves the final CSV file used by `src/train_ml.py`.
"""

from pathlib import Path

from datasets import concatenate_datasets, load_dataset

from data_preprocessing import preprocess_dataframe


def map_label_to_sentiment(label: int) -> str:
    """Convert numeric IMDB labels into readable sentiment labels."""
    label_map = {0: "negative", 1: "positive"}
    if label not in label_map:
        raise ValueError(f"Unexpected label value: {label}")
    return label_map[label]


def sample_by_label(split_dataset, label_value: int, sample_size: int):
    """Return a fixed-size subset for one label from a split.

    Why this exists:
    - The IMDB split is ordered in a way that can make "take first N rows"
      accidentally pick only one class.
    - We explicitly sample each class so the final training data is balanced.
    """
    label_subset = split_dataset.filter(lambda example: example["label"] == label_value)
    if len(label_subset) < sample_size:
        raise ValueError(
            f"Not enough examples for label {label_value}. "
            f"Requested {sample_size}, found {len(label_subset)}."
        )
    return label_subset.shuffle(seed=42).select(range(sample_size))


def main() -> None:
    """Download, preprocess, and save the IMDB subset for training."""
    output_path = Path("data/processed/processed_reviews.csv")

    # 1) Load the IMDB dataset from Hugging Face.
    imdb_dataset = load_dataset("stanfordnlp/imdb")

    # 2) Build a balanced subset from each split.
    # Balanced sampling is important here because Logistic Regression needs
    # at least two classes and learns better when classes are represented fairly.
    train_negative = sample_by_label(imdb_dataset["train"], label_value=0, sample_size=1000)
    train_positive = sample_by_label(imdb_dataset["train"], label_value=1, sample_size=1000)

    test_negative = sample_by_label(imdb_dataset["test"], label_value=0, sample_size=250)
    test_positive = sample_by_label(imdb_dataset["test"], label_value=1, sample_size=250)

    # 3) Combine train and test subsets.
    combined_subset = concatenate_datasets(
        [train_negative, train_positive, test_negative, test_positive]
    )

    # 4) Shuffle the combined dataset so labels are mixed.
    combined_subset = combined_subset.shuffle(seed=42)

    # 5) Convert to pandas and map labels (0->negative, 1->positive).
    df = combined_subset.to_pandas()
    df["sentiment"] = df["label"].apply(map_label_to_sentiment)

    # 6) Rename text column to 'review' and keep needed columns.
    df = df.rename(columns={"text": "review"})
    df = df[["review", "sentiment"]]

    # 7) Apply existing preprocessing to create 'cleaned_review'.
    processed_df = preprocess_dataframe(df, text_column="review", label_column="sentiment")

    # 8) Save to the path expected by src/train_ml.py.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    # 9) Print final dataset size and label distribution.
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
