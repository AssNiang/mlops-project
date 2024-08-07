import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from src.make_dataset import load_data

st.page_link("streamlit-deploy.py", label="Home", icon="🏠")
st.page_link("pages/1_classify.py", label="Classify", icon="1️⃣")
st.page_link("pages/2_dataset.py", label="Dataset", icon="2️⃣", disabled=True)

# load data
df = load_data("data/ecommerceDataset.csv")

# print the dataset
st.write("## Dataset loaded")
st.dataframe(df)
