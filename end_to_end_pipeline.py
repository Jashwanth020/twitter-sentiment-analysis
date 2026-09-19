import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

# Ensure NLTK resources are available
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# ---------------------------------------------------------
# STEP 4: Load and Inspect Data
# ---------------------------------------------------------
print("Loading data...")
raw_data_path = "data/raw/Tweets.csv"
if not os.path.exists(raw_data_path):
    raise FileNotFoundError(f"{raw_data_path} not found.")

df = pd.read_csv(raw_data_path)
print(f"Dataset shape: {df.shape}")
print("Target distribution:")
print(df["airline_sentiment"].value_counts(normalize=True) * 100)

# ---------------------------------------------------------
# STEP 5: Perform EDA (Save plots)
# ---------------------------------------------------------
print("Performing EDA...")
os.makedirs("eda_plots", exist_ok=True)

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="airline_sentiment", order=["negative", "neutral", "positive"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.savefig("eda_plots/sentiment_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
df["negativereason"].value_counts().plot(kind="barh")
plt.title("Reasons for Negative Sentiment")
plt.xlabel("Number of Tweets")
plt.tight_layout()
plt.savefig("eda_plots/negative_reasons.png")
plt.close()

# ---------------------------------------------------------
# STEP 6 & 7: Clean, Validate, and Save Modeling Dataset
# ---------------------------------------------------------
print("Creating modeling dataset...")
model_df = df[["text", "airline_sentiment"]].copy()
model_df = model_df.rename(columns={"airline_sentiment": "sentiment"})
model_df = model_df.dropna(subset=["text", "sentiment"])
model_df = model_df.drop_duplicates(subset=["text"])
model_df.to_csv("data/processed/sentiment_dataset.csv", index=False)
print(f"Processed dataset shape: {model_df.shape}")

# ---------------------------------------------------------
# STEP 8: Create Train/Test Split
# ---------------------------------------------------------
X = model_df["text"]
y = model_df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------------------------------------------------------
# STEP 9: Build Text Preprocessing
# ---------------------------------------------------------
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation for classical models
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

X_train_clean = X_train.apply(preprocess_text)
X_test_clean = X_test.apply(preprocess_text)

# ---------------------------------------------------------
# STEP 10: Implement VADER Baseline
# ---------------------------------------------------------
print("Evaluating VADER Baseline...")
sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

vader_predictions = X_test.apply(vader_sentiment)
vader_macro_f1 = f1_score(y_test, vader_predictions, average="macro")
vader_weighted_f1 = f1_score(y_test, vader_predictions, average="weighted")
vader_acc = accuracy_score(y_test, vader_predictions)

# ---------------------------------------------------------
# STEPS 11-13: Implement ML Models
# ---------------------------------------------------------
print("Training Classical ML Models...")

tfidf_params = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True
}

# 1. Naive Bayes
nb_model = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_params)),
    ("classifier", MultinomialNB())
])
nb_model.fit(X_train_clean, y_train)
nb_predictions = nb_model.predict(X_test_clean)

# 2. Logistic Regression
lr_model = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_params)),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
lr_model.fit(X_train_clean, y_train)
lr_predictions = lr_model.predict(X_test_clean)

# 3. Linear SVM
svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_params)),
    ("classifier", LinearSVC(class_weight="balanced"))
])
svm_model.fit(X_train_clean, y_train)
svm_predictions = svm_model.predict(X_test_clean)

# ---------------------------------------------------------
# STEPS 14-15: Evaluate and Compare Models
# ---------------------------------------------------------
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    return {"Model": model_name, "Accuracy": acc, "Macro F1": macro_f1, "Weighted F1": weighted_f1}

results = [
    evaluate_model(y_test, vader_predictions, "VADER"),
    evaluate_model(y_test, nb_predictions, "Naive Bayes"),
    evaluate_model(y_test, lr_predictions, "Logistic Regression"),
    evaluate_model(y_test, svm_predictions, "Linear SVM")
]

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Confusion Matrix for best model (assuming Linear SVM or LR usually performs best, we'll plot for SVM)
best_model_name = results_df.loc[results_df["Macro F1"].idxmax()]["Model"]
print(f"\nBest model by Macro F1 is: {best_model_name}")

if best_model_name == "Linear SVM":
    best_predictions = svm_predictions
    best_pipeline = svm_model
elif best_model_name == "Logistic Regression":
    best_predictions = lr_predictions
    best_pipeline = lr_model
elif best_model_name == "Naive Bayes":
    best_predictions = nb_predictions
    best_pipeline = nb_model
else:
    best_predictions = vader_predictions
    best_pipeline = None

# Confusion matrix plot
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best_predictions, labels=["negative", "neutral", "positive"])
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["negative", "neutral", "positive"], yticklabels=["negative", "neutral", "positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{best_model_name} Confusion Matrix")
plt.savefig(f"eda_plots/{best_model_name.replace(' ', '_')}_confusion_matrix.png")
plt.close()

# ---------------------------------------------------------
# STEP 16: Error Analysis
# ---------------------------------------------------------
print("\nPerforming Error Analysis on best model...")
errors = model_df.loc[X_test.index].copy()
errors["prediction"] = best_predictions
errors["correct"] = (errors["sentiment"] == errors["prediction"])
misclassified = errors[~errors["correct"]]
print("Sample misclassifications:")
print(misclassified[["text", "sentiment", "prediction"]].head(5))

# ---------------------------------------------------------
# STEPS 17-18: Serialize Best Model
# ---------------------------------------------------------
if best_pipeline:
    # Retrain on full data? The instruction says to just dump the pipeline used.
    # The instructions: "The saved pipeline should contain: preprocessing/TF-IDF, classifier"
    # Wait, our clean_text/preprocess_text is NOT in the scikit-learn Pipeline object itself.
    # We need a custom transformer or we have to call preprocess_text before predict.
    # Alternatively, we can use a FunctionTransformer or include it in Streamlit.
    # Let's save the model and in Streamlit we'll call preprocess_text.
    
    # Save the model
    model_path = "models/sentiment_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved {best_model_name} model to {model_path}")
