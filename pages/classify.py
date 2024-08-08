"""Streamlit app for classifying consumer goods based on description."""

import requests
import sys
from pathlib import Path
import pickle
import streamlit as st

sys.path.append(str(Path.cwd()))

#pylint: disable=wrong-import-position

from src.preprocess import text_normalizer
from settings.params import MODEL_PARAMS

#pylint: enable=wrong-import-position

# Navigation buttons
st.page_link("streamlit_deploy.py", label="Home", icon="🏠")
st.page_link("pages/classify.py", label="Classify", icon="1️⃣", disabled=True)
st.page_link("pages/dataset.py", label="Dataset", icon="2️⃣")

# Title
st.title("PROJET | Classifiez des biens de consommation")
st.write("EPT/GIT/DIC3/MLOPS")

# Input the description
st.write("### Description of the article")
article_description = st.text_area("Enter the description of the article")

# URL of the FastAPI server
URL = "http://127.0.0.1:8000/predict"

# Sending POST request with timeout
try:
    response = requests.post(
        URL,
        params={"article_description": article_description},
        headers={"accept": "application/json"},
        timeout=10  # Adding timeout to prevent indefinite hanging
    )
    response.raise_for_status()  # Raise HTTPError for bad responses
    predicted_label = response.json().get("predicted_label")
except Exception as e:
    st.error(f"The API is unavailable due to : {e}")
    
    # Directly calling the model
    # Load the classifier
    with open("models/classifier.pkl", "rb") as pickle_in:
        classifier = pickle.load(pickle_in)
    # Load the vectorizer
    with open("models/tfidf_vectorizer.pkl", "rb") as pickle_in_vect:
        tfidf_vectorizer = pickle.load(pickle_in_vect)
    # Normalize the input text
    text_norm = text_normalizer(article_description)
    # Transform the normalized text using the loaded TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([text_norm])
    # Predict the label index using the classifier
    predicted_label_index = classifier.predict(text_tfidf)[0]
    # Map the label index to the actual label
    predicted_label = (
        ""
        if article_description == ""
        else MODEL_PARAMS["TARGET_LABELS"][predicted_label_index]
    )

# Display the prediction result
st.write(f"### Predicted tag : {predicted_label if predicted_label else 'No prediction available'}")
