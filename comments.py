import nltk
nltk.download('punkt')
nltk.download('stopwords')

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load spaCy English model
import en_core_web_sm
nlp = en_core_web_sm.load()

# Load dataset (this file must contain 'comment' and 'sentiment' columns)
df = pd.read_csv('synthetic_reel_comments.csv')

# Make sure sentiment column is lowercase to match mapping
df['sentiment'] = df['sentiment'].str.lower()

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'<.*?>', '', text)               # Remove HTML tags
    text = text.lower()                             # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)            # Remove punctuation/numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    doc = nlp(' '.join(tokens))                     # Lemmatization
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)

# Drop null rows and sample
df = df.dropna(subset=['comment', 'sentiment'])
df = df.sample(n=min(500, len(df))).copy()  # Limit to 500 rows

# Clean the comments
df['cleaned_comment'] = df['comment'].apply(preprocess_text)

# Encode sentiment: Positive = 1, Negative = 0, drop Neutral for binary classification
df = df[df['sentiment'].isin(['positive', 'negative'])]
df['sentiment_numeric'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Features and labels
X = df['cleaned_comment']
y = df['sentiment_numeric']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train_tfidf, y_train)

# Predict function
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Sample predictions for Instagram Reel comments
print(predict_sentiment("This reel was so amazing, I shared it instantly!"))
print(predict_sentiment("Boring content, didn’t enjoy it at all."))
print(predict_sentiment("It was okay I guess, nothing special."))

# Evaluation
y_pred = model.predict(X_test_tfidf)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
