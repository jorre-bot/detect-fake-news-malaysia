import os
import joblib
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import streamlit as st

def debug_nltk_paths():
    """Debug function to print NLTK paths and data"""
    st.write("NLTK Data Paths:", nltk.data.path)
    try:
        punkt_loc = nltk.data.find('tokenizers/punkt')
        st.write("Punkt location:", punkt_loc)
    except LookupError:
        st.write("Punkt not found in NLTK paths")

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    try:
        # Set NLTK data path to the app's directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        nltk_data_dir = os.path.join(app_dir, 'nltk_data')
        
        # Create directory if it doesn't exist
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Add the nltk_data directory to NLTK's data path
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
        
        # Try to find punkt, download if not found
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', download_dir=nltk_data_dir)
            
        # Verify punkt is available
        nltk.data.find('tokenizers/punkt')
        
    except Exception as e:
        st.error(f"Error setting up NLTK data: {str(e)}")
        debug_nltk_paths()
        raise e

def preprocess_text(text):
    """Simple text preprocessing"""
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")
        
    try:
        # Ensure NLTK data is available before preprocessing
        ensure_nltk_data()
        
        # Convert to lowercase
        text = text.lower()
        
        # Debug tokenization
        st.write("Preprocessing text:", text[:100] + "..." if len(text) > 100 else text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Debug tokens
        st.write("First few tokens:", tokens[:10])
        
        # Join back to string
        processed_text = ' '.join(tokens)
        
        if not processed_text:
            raise ValueError("Preprocessing resulted in empty text")
            
        return processed_text
        
    except Exception as e:
        st.error(f"Text preprocessing error: {str(e)}")
        raise e

@st.cache_resource
def load_model():
    """Load the model with caching"""
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(app_dir, 'best_fake_news_model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = joblib.load(model_path)
        
        # Verify model has required methods
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise AttributeError("Loaded model doesn't have required prediction methods")
            
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

def predict_news(text):
    """Predict if news is real or fake"""
    try:
        if not text or not text.strip():
            raise ValueError("Please enter some text to analyze")
            
        # Load model
        model = load_model()
        st.write("Model loaded successfully")
        
        # Preprocess text
        processed_text = preprocess_text(text)
        st.write("Text preprocessed successfully")
        
        # Debug processed text
        st.write("Processed text sample:", processed_text[:100] + "..." if len(processed_text) > 100 else processed_text)
        
        # Make prediction
        prediction = model.predict([processed_text])[0]
        probabilities = model.predict_proba([processed_text])[0]
        
        # Debug prediction values
        st.write("Raw prediction:", prediction)
        st.write("Raw probabilities:", probabilities)
        
        # Get confidence score
        confidence = float(max(probabilities))
        confidence = min(confidence * 100, 100.0)  # Convert to percentage and cap at 100%
        
        result = {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': confidence
        }
        
        st.write("Prediction result:", result)
        return result
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return {
            'prediction': 'Error',
            'confidence': 0.0
        }
