import os
import joblib
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import streamlit as st

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    nltk.data.path.append(nltk_data_dir)
    
    required_resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

# Ensure NLTK data is available
ensure_nltk_data()

def preprocess_text(text):
    """Simple text preprocessing"""
    try:
        # Convert to string and lowercase
        text = str(text).lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Join back to string
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Text preprocessing error: {str(e)}")
        return text

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
        # Ensure NLTK data is available
        ensure_nltk_data()
        
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
