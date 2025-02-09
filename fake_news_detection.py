import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
df = pd.read_csv("news.csv")  # Make sure the dataset file is in your project folder
print(df.head())

# Data Preprocessing
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

df['text'] = df['text'].apply(preprocess_text)

# Split Dataset
X = df['text']
y = df['label']  # Assuming the dataset has 'label' column with 'FAKE' or 'REAL'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Convert Text to Numerical Data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluate the Model
y_pred = model.predict(X_test_tfidf)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {score*100:.2f}%')

# Test with Custom Input
def predict_news(news_text):
    news_tfidf = vectorizer.transform([news_text])
    prediction = model.predict(news_tfidf)
    return prediction[0]

test_text = "Breaking: Scientists discover a new planet similar to Earth!"
print(f"Prediction: {predict_news(test_text)}")
