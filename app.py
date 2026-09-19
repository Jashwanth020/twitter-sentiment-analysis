import streamlit as st
import joblib
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK resources are available
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="🐦")
st.title("Twitter Sentiment Analyzer")

@st.cache_resource
def load_model():
    return joblib.load("models/sentiment_model.joblib")

model = load_model()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

st.header("Single Tweet Prediction")
text = st.text_area("Enter a tweet:")

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter a tweet.")
    else:
        preprocessed = preprocess_text(text)
        prediction = model.predict([preprocessed])[0]
        st.success(f"Predicted Sentiment: {prediction}")
        # Note: If predict_proba is needed and supported, we could display it here.

st.markdown("---")
st.header("Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV containing a 'text' column", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    if "text" not in batch_df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        with st.spinner("Analyzing batch..."):
            batch_df["cleaned_text"] = batch_df["text"].apply(preprocess_text)
            batch_df["sentiment"] = model.predict(batch_df["cleaned_text"])
            batch_df = batch_df.drop(columns=["cleaned_text"])
        
        st.success("Batch analysis complete!")
        st.dataframe(batch_df.head(50))
        
        csv_data = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions",
            data=csv_data,
            file_name="batch_predictions.csv",
            mime="text/csv",
        )
