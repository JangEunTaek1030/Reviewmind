"""
Train a beginner-friendly Transformer baseline for sentiment classification.

This script reads the processed ReviewMind dataset, fine-tunes a small Hugging Face
Transformer model (DistilBERT by default), evaluates it on a held-out test set,
and saves the trained model artifacts.

Expected input file:
    data/processed/processed_reviews.csv

Expected columns in the CSV:
    - cleaned_review
    - sentiment
"""

from pathlib import Path
import argparse

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"
INPUT_PATH = Path("data/processed/processed_reviews.csv")
OUTPUT_DIR_BASELINE = Path("outputs/models/transformer_baseline")
OUTPUT_DIR_DEBUG = Path("outputs/models/transformer_debug")
TEXT_COLUMN = "cleaned_review"
LABEL_COLUMN = "sentiment"
FULL_MAX_LENGTH = 256
DEBUG_MAX_LENGTH = 128


def load_processed_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate the processed dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [TEXT_COLUMN, LABEL_COLUMN]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in CSV: {column}")

    df = df.dropna(subset=required_columns).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    return df


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, dict, dict]:
    """Map sentiment labels to integer ids for Transformer training."""
    unique_labels = sorted(labels.unique())

    if len(unique_labels) != 2:
        raise ValueError(
            "This baseline expects binary sentiment labels. "
            f"Found {len(unique_labels)} labels: {unique_labels}"
        )

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    encoded_labels = labels.map(label2id).to_numpy()
    return encoded_labels, label2id, id2label


def tokenize_batch(examples: dict, tokenizer: AutoTokenizer, max_length: int) -> dict:
    """Tokenize a batch of texts."""
    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
        max_length=max_length,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for full or debug training."""
    parser = argparse.ArgumentParser(
        description="Train the ReviewMind DistilBERT sentiment baseline."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Run a fast debug training mode with a small balanced subset "
            "for local pipeline testing."
        ),
    )
    return parser.parse_args()


def build_debug_subset(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Build a small class-balanced subset for quick local debugging."""
    target_train = 300
    target_test = 100
    target_total = target_train + target_test

    class_counts = df[LABEL_COLUMN].value_counts()
    n_classes = len(class_counts)
    samples_per_class = target_total // n_classes

    subset_parts = []
    for label in sorted(class_counts.index):
        label_df = df[df[LABEL_COLUMN] == label]
        take_n = min(samples_per_class, len(label_df))
        subset_parts.append(label_df.sample(n=take_n, random_state=random_state))

    subset_df = pd.concat(subset_parts, ignore_index=True)
    return subset_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def main() -> None:
    """Run training, evaluation, and model saving."""
    args = parse_args()
    is_debug = args.debug

    # 1) Load data
    df = load_processed_data(INPUT_PATH)

    if is_debug:
        print("\n" + "!" * 72)
        print("WARNING: DEBUG MODE ENABLED")
        print(
            "Debug metrics are only for pipeline testing and must not be reported "
            "as final model performance."
        )
        print("!" * 72 + "\n")
        df = build_debug_subset(df, random_state=42)
        max_length = DEBUG_MAX_LENGTH
        output_dir = OUTPUT_DIR_DEBUG
        train_batch_size = 8
        eval_batch_size = 8
        num_train_epochs = 1
        test_size = 0.25
    else:
        max_length = FULL_MAX_LENGTH
        output_dir = OUTPUT_DIR_BASELINE
        train_batch_size = 16
        eval_batch_size = 16
        num_train_epochs = 2
        test_size = 0.2

    # 2) Encode labels from text/categorical values to integer ids
    y_encoded, label2id, id2label = encode_labels(df[LABEL_COLUMN])
    X = df[TEXT_COLUMN]

    print("Dataset Diagnostics")
    print("-" * 30)
    print(f"Total samples: {len(df)}")
    print(f"Label mapping: {label2id}")

    # 3) Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples    : {len(X_test)}")

    # 4) Build Hugging Face datasets from pandas splits
    train_df = pd.DataFrame({TEXT_COLUMN: X_train.values, "label": y_train})
    test_df = pd.DataFrame({TEXT_COLUMN: X_test.values, "label": y_test})

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    # 5) Load tokenizer and tokenize data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = train_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        desc="Tokenizing training split",
    )
    tokenized_test = test_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        desc="Tokenizing test split",
    )

    # 6) Load model for binary classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7) Define metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
            "accuracy"
        ]
        precision = precision_metric.compute(
            predictions=predictions,
            references=labels,
            average="binary",
        )["precision"]
        recall = recall_metric.compute(
            predictions=predictions,
            references=labels,
            average="binary",
        )["recall"]
        f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="binary",
        )["f1"]

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # 8) Configure trainer
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",  # keep setup simple for beginners
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9) Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    print("\nModel Evaluation Results")
    print("-" * 30)
    print(f"Accuracy : {eval_results['eval_accuracy']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall   : {eval_results['eval_recall']:.4f}")
    print(f"F1-score : {eval_results['eval_f1']:.4f}")

    # 10) Save final model + tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved trained model and tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()
