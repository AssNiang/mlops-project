"""Streamlit app for displaying the dataset."""

import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path.cwd()))

#pylint: disable=wrong-import-position

from src.make_dataset import load_data

#pylint: enable=wrong-import-position

# Navigation buttons
st.page_link("streamlit_deploy.py", label="Home", icon="🏠")
st.page_link("pages/classify.py", label="Classify", icon="1️⃣")
st.page_link("pages/dataset.py", label="Dataset", icon="2️⃣", disabled=True)

# Load data
DATA_FILE_PATH = "data/ecommerceDataset.csv"
df = load_data(DATA_FILE_PATH)

# Display the dataset
st.write("## Dataset loaded")
st.dataframe(df)
