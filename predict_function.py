import os
import joblib
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import streamlit as st

# Download NLTK data at startup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to string and lowercase
    text = str(text).lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Join back to string
    return ' '.join(tokens)

@st.cache_resource
def load_model():
    """Load the model with caching"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'best_fake_news_model.pkl')
        return joblib.load(model_path)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_news(text):
    """Predict if news is real or fake"""
    try:
        # Load model (cached)
        model = load_model()
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Make prediction
        prediction = model.predict([processed_text])[0]
        probabilities = model.predict_proba([processed_text])[0]
        
        # Get confidence score
        confidence = float(max(probabilities))
        
        return {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': confidence
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return {
            'prediction': 'Error',
            'confidence': 0.0
        }
