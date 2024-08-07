import streamlit as st
import requests

# Navigation buttons
st.page_link("streamlit-deploy.py", label="Home", icon="🏠")
st.page_link("pages/1_classify.py", label="Classify", icon="1️⃣", disabled=True)
st.page_link("pages/2_dataset.py", label="Dataset", icon="2️⃣")

# Title
st.title("PROJET | Classifiez des biens de consommation")
st.write("EPT/GIT/DIC3/MLOPS")

# Input the description
st.write("### Description of the article")
article_description = st.text_area("Enter the description of the article")

# URL of the FastAPI server
url = "http://127.0.0.1:8000/predict"

# Sending POST request
response = requests.post(
    url,
    params={"article_description": article_description},
    headers={"accept": "application/json"},
)

predicted_label = (
    response.json()["predicted_label"] if response.status_code == 200 else None
)

# Display the prediction result
st.write(f"### Predicted tag : {predicted_label}")
