# Twitter Sentiment Analysis

End-to-End Twitter Sentiment Classification pipeline using classical NLP, machine learning (Scikit-Learn), and Streamlit.

## Project Overview

This project builds a sentiment analysis system on the Twitter US Airline Sentiment dataset. It implements data cleaning, exploratory data analysis (EDA), and compares several classical modeling approaches:

- **VADER** (Rule-based Baseline)
- **Naive Bayes** with TF-IDF
- **Logistic Regression** with TF-IDF
- **Linear SVM** with TF-IDF

The best-performing model (based on Macro F1 score) is serialized and served using a Streamlit application that supports real-time single-tweet analysis and batch prediction via CSV upload.

## Pipeline and Model Architecture

1. **Data Preprocessing Pipeline:**
    - Converts text to lowercase.
    - Removes URLs, mentions (`@user`), retweets (`RT`), and punctuation.
    - Removes excessive whitespaces.
    - Tokenizes and lemmatizes using NLTK (WordNetLemmatizer).
    - Applies `TfidfVectorizer` (unigrams & bigrams, max features = 50,000, sublinear TF scaling).

2. **Modeling Pipeline:**
    - The cleaned text is passed into the `TfidfVectorizer` which converts it to a numerical sparse matrix.
    - This matrix is fed into a classifier (Linear SVM, Logistic Regression, or Naive Bayes).
    - Class weights are set to 'balanced' to handle class imbalances for models that support it.

## Instructions for Running the Code

### 1. Setup Environment
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```
*(NLTK dependencies will be downloaded automatically when running the scripts)*

### 2. Run the End-to-End Pipeline
To run the full end-to-end data processing, model training, evaluation, and serialization:
```bash
python end_to_end_pipeline.py
```
This will:
- Read `data/raw/Tweets.csv`
- Perform EDA and save visual plots in `eda_plots/`
- Process the dataset and save it to `data/processed/sentiment_dataset.csv`
- Evaluate models (Accuracy, Precision, Recall, Macro F1, Weighted F1)
- Display the confusion matrix for the best model and save it in `eda_plots/`
- Serialize the best trained model to `models/sentiment_model.joblib`

### 3. Run the Streamlit Application
To interact with the deployed model via a web UI:
```bash
streamlit run app.py
```
This will launch a local server. You can:
- Enter a single tweet and predict sentiment.
- Upload a CSV containing a `text` column to perform batch analysis, and download the results.

## Project Structure
- `data/` : Raw and processed data
- `eda_plots/` : Output directory for EDA charts and Confusion Matrix
- `models/` : Saved `.joblib` model binaries
- `src/` : (Optional) Extended source modules
- `tests/` : Unit tests for text preprocessing
- `end_to_end_pipeline.py` : Main ML training script
- `app.py` : Streamlit inference application
