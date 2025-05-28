import joblib
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

def preprocess_text(text):
    import re
    # Convert to string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove numbers and special characters but keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def predict_news(text):
    try:
        # Load the model using joblib
        model = joblib.load('best_fake_news_model.pkl')
        
        # Preprocess the text
        tokens = word_tokenize(text.lower())
        
        # Make prediction
        prediction = model.predict([text])
        probabilities = model.predict_proba([text])
        
        # Get confidence score
        confidence = np.max(probabilities)
        
        # Return result
        return {
            'prediction': 'Real' if prediction[0] == 1 else 'Fake',
            'confidence': float(confidence)
        }
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")
