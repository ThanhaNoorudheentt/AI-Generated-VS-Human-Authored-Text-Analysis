import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import os

def train():
    print("Loading dataset: Hello-SimpleAI/hc3 (default)...")
    # Load default subset of HC3
    ds = load_dataset("Hello-SimpleAI/hc3", "default", split="train")
    
    # Preprocessing: Flatten and prepare data
    print("Preprocessing data...")
    data = []
    for item in ds:
        # Human answers
        for h in item['human_answers']:
            if len(h.split()) >= 10:
                data.append({'text': h, 'label': 0})
        # ChatGPT answers
        for c in item['chatgpt_answers']:
            if len(c.split()) >= 10:
                data.append({'text': c, 'label': 1})
    
    df = pd.DataFrame(data)
    
    # Take a smaller subset for speed (5000 samples)
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)
    
    print(f"Training on {len(df)} samples...")
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['text'])
    y = df['label']
    
    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save artifacts
    print("Saving model and vectorizer...")
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    joblib.dump(model, 'lr_model.joblib')
    print("Successfully saved tfidf_vectorizer.joblib and lr_model.joblib")

if __name__ == "__main__":
    train()
