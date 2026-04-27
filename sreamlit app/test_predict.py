import joblib

def test():
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('lr_model.joblib')
    
    ai_text = "The City of Light has been the heart of Western culture for centuries. A few things that make it endlessly fascinating: 1. Architecture: From the Gothic majesty of Notre-Dame to the iron lattice of the Eiffel Tower."
    human_text = "Honestly, I think Paris is okay but it's super expensive and way too many tourists everywhere. The bread was cool though."
    
    # AI Test
    X_ai = tfidf.transform([ai_text])
    pred_ai = model.predict(X_ai)[0]
    prob_ai = model.predict_proba(X_ai)[0][1]
    print(f"AI Text Prediction: {'AI' if pred_ai == 1 else 'Human'} (AI Prob: {prob_ai:.2%})")
    
    # Human Test
    X_human = tfidf.transform([human_text])
    pred_human = model.predict(X_human)[0]
    prob_human = model.predict_proba(X_human)[0][0]
    print(f"Human Text Prediction: {'Human' if pred_human == 0 else 'AI'} (Human Prob: {prob_human:.2%})")

if __name__ == "__main__":
    test()
