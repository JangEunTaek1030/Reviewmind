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
OUTPUT_DIR = Path("outputs/models/transformer_baseline")
TEXT_COLUMN = "cleaned_review"
LABEL_COLUMN = "sentiment"
MAX_LENGTH = 256


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


def tokenize_batch(examples: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenize a batch of texts."""
    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def main() -> None:
    """Run training, evaluation, and model saving."""
    # 1) Load data
    df = load_processed_data(INPUT_PATH)

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
        test_size=0.2,
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
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        desc="Tokenizing training split",
    )
    tokenized_test = test_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer),
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
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Saved trained model and tokenizer to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
