"""Streamlit app for classifying consumer goods based on description."""

import streamlit as st
import requests

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
except requests.RequestException as e:
    st.error(f"An error occurred: {e}")

# Display the prediction result
st.write(f"### Predicted tag : {predicted_label if predicted_label else 'No prediction available'}")
