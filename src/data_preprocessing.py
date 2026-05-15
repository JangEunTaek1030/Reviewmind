"""
Data preprocessing utilities for ReviewMind.

This module contains helper functions for cleaning and preparing
user review text for machine learning and deep learning models.
"""

import re
from typing import Optional

import pandas as pd


def clean_text(text: Optional[str]) -> str:
    """
    Clean a single review text.

    Parameters
    ----------
    text : Optional[str]
        Raw review text.

    Returns
    -------
    str
        Cleaned review text.
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text)
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "review",
    label_column: str = "sentiment",
) -> pd.DataFrame:
    """
    Preprocess a review dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
    text_column : str
        Name of the column containing review text.
    label_column : str
        Name of the column containing sentiment labels.

    Returns
    -------
    pd.DataFrame
        Processed dataframe with cleaned review text and labels.
    """
    required_columns = [text_column, label_column]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    processed_df = df[[text_column, label_column]].copy()
    processed_df = processed_df.dropna(subset=[text_column, label_column])

    processed_df["cleaned_review"] = processed_df[text_column].apply(clean_text)

    processed_df = processed_df[processed_df["cleaned_review"] != ""]

    return processed_df


if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "review": [
                "This product is AMAZING!!!",
                "Terrible quality... I will not buy again.",
                None,
            ],
            "sentiment": ["positive", "negative", "neutral"],
        }
    )

    processed_data = preprocess_dataframe(sample_data)
    print(processed_data)
