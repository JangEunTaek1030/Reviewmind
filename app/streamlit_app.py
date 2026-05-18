from pathlib import Path
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

if not MODEL_PATH.exists():
    st.warning(
        "I couldn't find the trained model file at `outputs/models/ml_baseline_model.pkl`.\n\n"
        "Please run these commands first:\n"
        "`python src/prepare_imdb_dataset.py`\n"
        "`python src/train_ml.py`"
    )
    st.stop()

model = load_model(MODEL_PATH)

review_text = st.text_area(
    "Movie review",
    placeholder="Type your review here...",
    height=180,
)

if st.button("Predict Sentiment"):
    if not review_text.strip():
        st.info("Please enter a movie review before predicting.")
    else:
        prediction = model.predict([review_text])[0]
        sentiment = "Positive ✅" if str(prediction) in {"1", "positive", "pos"} else "Negative ❌"
        st.subheader("Predicted Sentiment")
        st.success(sentiment)
