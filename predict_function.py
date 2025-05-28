import os
import joblib
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import streamlit as st

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    try:
        # Set NLTK data path to the app's directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(app_dir, 'nltk_data')
        
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Add the nltk_data directory to NLTK's data path
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
        
        # Download required NLTK resources
        nltk.download('punkt', download_dir=nltk_data_dir)
        
        # Verify that punkt is available
        nltk.data.find('tokenizers/punkt')
    except Exception as e:
        st.error(f"Error setting up NLTK data: {str(e)}")
        raise e

def preprocess_text(text):
    """Simple text preprocessing"""
    try:
        # Ensure NLTK data is available before preprocessing
        ensure_nltk_data()
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Join back to string
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Text preprocessing error: {str(e)}")
        raise e

@st.cache_resource
def load_model():
    """Load the model with caching"""
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_fake_news_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

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
        
        # Ensure confidence is in reasonable range
        confidence = min(confidence * 100, 100.0)
        
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
