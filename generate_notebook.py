import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Twitter Sentiment Analysis\n",
                "End-to-End pipeline for sentiment classification up to model saving."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import re\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import nltk\n",
                "from nltk.sentiment import SentimentIntensityAnalyzer\n",
                "from nltk.tokenize import word_tokenize\n",
                "from nltk.stem import WordNetLemmatizer\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.naive_bayes import MultinomialNB\n",
                "from sklearn.svm import LinearSVC\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score, precision_score, recall_score, f1_score,\n",
                "    classification_report, confusion_matrix\n",
                ")\n",
                "import joblib\n",
                "\n",
                "# Ensure NLTK resources are available\n",
                "nltk.download(\"punkt\", quiet=True)\n",
                "nltk.download(\"stopwords\", quiet=True)\n",
                "nltk.download(\"wordnet\", quiet=True)\n",
                "nltk.download(\"vader_lexicon\", quiet=True)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Load and Inspect Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_data_path = \"data/raw/Tweets.csv\"\n",
                "df = pd.read_csv(raw_data_path)\n",
                "print(f\"Dataset shape: {df.shape}\")\n",
                "print(\"\\nTarget distribution:\")\n",
                "print(df[\"airline_sentiment\"].value_counts(normalize=True) * 100)\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Exploratory Data Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 5))\n",
                "sns.countplot(data=df, x=\"airline_sentiment\", order=[\"negative\", \"neutral\", \"positive\"])\n",
                "plt.title(\"Sentiment Distribution\")\n",
                "plt.xlabel(\"Sentiment\")\n",
                "plt.ylabel(\"Number of Tweets\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 6))\n",
                "df[\"negativereason\"].value_counts().plot(kind=\"barh\")\n",
                "plt.title(\"Reasons for Negative Sentiment\")\n",
                "plt.xlabel(\"Number of Tweets\")\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Steps 6 & 7: Clean, Validate, and Save Modeling Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model_df = df[[\"text\", \"airline_sentiment\"]].copy()\n",
                "model_df = model_df.rename(columns={\"airline_sentiment\": \"sentiment\"})\n",
                "model_df = model_df.dropna(subset=[\"text\", \"sentiment\"])\n",
                "model_df = model_df.drop_duplicates(subset=[\"text\"])\n",
                "\n",
                "os.makedirs(\"data/processed\", exist_ok=True)\n",
                "model_df.to_csv(\"data/processed/sentiment_dataset.csv\", index=False)\n",
                "print(f\"Processed dataset shape: {model_df.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 8: Create Train/Test Split"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X = model_df[\"text\"]\n",
                "y = model_df[\"sentiment\"]\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.20, random_state=42, stratify=y\n",
                ")\n",
                "print(f\"Training set size: {len(X_train)}\")\n",
                "print(f\"Testing set size: {len(X_test)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 9: Build Text Preprocessing"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "lemmatizer = WordNetLemmatizer()\n",
                "\n",
                "def clean_text(text):\n",
                "    text = text.lower()\n",
                "    text = re.sub(r\"http\\S+|www\\S+\", \"\", text)\n",
                "    text = re.sub(r\"@\\w+\", \"\", text)\n",
                "    text = re.sub(r\"\\brt\\b\", \"\", text)\n",
                "    text = re.sub(r\"[^\\w\\s]\", \"\", text) # Remove punctuation for classical models\n",
                "    text = re.sub(r\"\\s+\", \" \", text)\n",
                "    return text.strip()\n",
                "\n",
                "def preprocess_text(text):\n",
                "    cleaned = clean_text(text)\n",
                "    tokens = word_tokenize(cleaned)\n",
                "    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]\n",
                "    return \" \".join(lemmatized)\n",
                "\n",
                "X_train_clean = X_train.apply(preprocess_text)\n",
                "X_test_clean = X_test.apply(preprocess_text)\n",
                "print(\"Sample preprocessed text:\", X_train_clean.iloc[0])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 10: Implement VADER Baseline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sia = SentimentIntensityAnalyzer()\n",
                "\n",
                "def vader_sentiment(text):\n",
                "    score = sia.polarity_scores(text)[\"compound\"]\n",
                "    if score >= 0.05:\n",
                "        return \"positive\"\n",
                "    elif score <= -0.05:\n",
                "        return \"negative\"\n",
                "    return \"neutral\"\n",
                "\n",
                "vader_predictions = X_test.apply(vader_sentiment)\n",
                "print(\"VADER Accuracy:\", accuracy_score(y_test, vader_predictions))\n",
                "print(\"VADER Macro F1:\", f1_score(y_test, vader_predictions, average=\"macro\"))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Steps 11-13: Train Classical ML Models"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tfidf_params = {\n",
                "    \"max_features\": 50000,\n",
                "    \"ngram_range\": (1, 2),\n",
                "    \"min_df\": 2,\n",
                "    \"max_df\": 0.95,\n",
                "    \"sublinear_tf\": True\n",
                "}\n",
                "\n",
                "# 1. Naive Bayes\n",
                "nb_model = Pipeline([\n",
                "    (\"tfidf\", TfidfVectorizer(**tfidf_params)),\n",
                "    (\"classifier\", MultinomialNB())\n",
                "])\n",
                "nb_model.fit(X_train_clean, y_train)\n",
                "nb_predictions = nb_model.predict(X_test_clean)\n",
                "\n",
                "# 2. Logistic Regression\n",
                "lr_model = Pipeline([\n",
                "    (\"tfidf\", TfidfVectorizer(**tfidf_params)),\n",
                "    (\"classifier\", LogisticRegression(max_iter=1000, class_weight=\"balanced\"))\n",
                "])\n",
                "lr_model.fit(X_train_clean, y_train)\n",
                "lr_predictions = lr_model.predict(X_test_clean)\n",
                "\n",
                "# 3. Linear SVM\n",
                "svm_model = Pipeline([\n",
                "    (\"tfidf\", TfidfVectorizer(**tfidf_params)),\n",
                "    (\"classifier\", LinearSVC(class_weight=\"balanced\"))\n",
                "])\n",
                "svm_model.fit(X_train_clean, y_train)\n",
                "svm_predictions = svm_model.predict(X_test_clean)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Steps 14-15: Evaluate and Compare Models"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate_model(y_true, y_pred, model_name):\n",
                "    acc = accuracy_score(y_true, y_pred)\n",
                "    macro_f1 = f1_score(y_true, y_pred, average=\"macro\")\n",
                "    weighted_f1 = f1_score(y_true, y_pred, average=\"weighted\")\n",
                "    return {\"Model\": model_name, \"Accuracy\": acc, \"Macro F1\": macro_f1, \"Weighted F1\": weighted_f1}\n",
                "\n",
                "results = [\n",
                "    evaluate_model(y_test, vader_predictions, \"VADER\"),\n",
                "    evaluate_model(y_test, nb_predictions, \"Naive Bayes\"),\n",
                "    evaluate_model(y_test, lr_predictions, \"Logistic Regression\"),\n",
                "    evaluate_model(y_test, svm_predictions, \"Linear SVM\")\n",
                "]\n",
                "\n",
                "results_df = pd.DataFrame(results)\n",
                "results_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 16: Error Analysis and Confusion Matrix"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "best_model_name = results_df.loc[results_df[\"Macro F1\"].idxmax()][\"Model\"]\n",
                "print(f\"Best model by Macro F1 is: {best_model_name}\")\n",
                "\n",
                "if best_model_name == \"Linear SVM\":\n",
                "    best_predictions = svm_predictions\n",
                "    best_pipeline = svm_model\n",
                "elif best_model_name == \"Logistic Regression\":\n",
                "    best_predictions = lr_predictions\n",
                "    best_pipeline = lr_model\n",
                "elif best_model_name == \"Naive Bayes\":\n",
                "    best_predictions = nb_predictions\n",
                "    best_pipeline = nb_model\n",
                "else:\n",
                "    best_predictions = vader_predictions\n",
                "    best_pipeline = None\n",
                "\n",
                "plt.figure(figsize=(6, 5))\n",
                "cm = confusion_matrix(y_test, best_predictions, labels=[\"negative\", \"neutral\", \"positive\"])\n",
                "sns.heatmap(cm, annot=True, fmt=\"d\", xticklabels=[\"negative\", \"neutral\", \"positive\"], yticklabels=[\"negative\", \"neutral\", \"positive\"])\n",
                "plt.xlabel(\"Predicted\")\n",
                "plt.ylabel(\"Actual\")\n",
                "plt.title(f\"{best_model_name} Confusion Matrix\")\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Steps 17-18: Serialize Best Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if best_pipeline:\n",
                "    os.makedirs(\"models\", exist_ok=True)\n",
                "    model_path = \"models/sentiment_model.joblib\"\n",
                "    joblib.dump(best_pipeline, model_path)\n",
                "    print(f\"Saved {best_model_name} pipeline to {model_path}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/End_to_End_Sentiment_Analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")
