from pathlib import Path
import re
import pickle

import streamlit as st

MODEL_PATH = Path("outputs/models/ml_baseline_model.pkl")


@st.cache_resource
def load_model(model_path: Path):
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


st.set_page_config(page_title="ReviewMind Demo", page_icon="🎬")
st.title("🎬 ReviewMind Sentiment Demo")
st.write(
    "Enter an English movie review and click **Predict Sentiment** to classify it as positive or negative."
)
st.caption(
    "This demo uses a TF-IDF + Logistic Regression baseline model. It may be unreliable for very short inputs, "
    "numeric-only ratings, sarcasm, or ambiguous reviews."
)

if not MODEL_PATH.exists():
    st.warning(
        "I couldn't find the trained model file at `outputs/models/ml_baseline_model.pkl`.\n\n"
        "Please run these commands first:\n"
        "`python src/prepare_imdb_dataset.py`\n"
        "`python src/train_ml.py`"
    )
    st.stop()

model = load_model(MODEL_PATH)


def extract_rating_out_of_ten(text: str):
    rating_patterns = [
        re.compile(r"\b([0-9](?:\.[0-9])?|10(?:\.0)?)\s*/\s*10\b"),
        re.compile(r"\b([0-9](?:\.[0-9])?|10(?:\.0)?)\s*out of\s*10\b", re.IGNORECASE),
    ]
    for pattern in rating_patterns:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None

review_text = st.text_area(
    "Movie review",
    placeholder="Type your review here...",
    height=180,
)

if st.button("Predict Sentiment"):
    if not review_text.strip():
        st.info("Please enter a movie review before predicting.")
    else:
        word_count = len(review_text.split())
        if word_count < 3:
            st.warning("This input is very short, so the prediction may be unreliable.")

        rating = extract_rating_out_of_ten(review_text)
        if rating is not None:
            if rating <= 4:
                st.info("Numeric rating note: this rating suggests **negative** sentiment.")
            elif rating >= 7:
                st.info("Numeric rating note: this rating suggests **positive** sentiment.")

        prediction = model.predict([review_text])[0]
        sentiment = "Positive ✅" if str(prediction) in {"1", "positive", "pos"} else "Negative ❌"

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([review_text])[0]
            if len(probabilities) > 1:
                confidence = float(max(probabilities))

        st.subheader("Predicted Sentiment")
        st.success(sentiment)
        if confidence is not None:
            st.write(f"Model confidence: {confidence * 100:.1f}%")
