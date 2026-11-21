import spacy
import dateparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit
import plotly
import pandas

# Verify spaCy model loading
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

print("Environment setup complete! All dependencies and spaCy model loaded successfully.")