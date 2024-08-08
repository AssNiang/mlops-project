"""Streamlit Deploy Module"""

import streamlit as st


st.page_link("streamlit_deploy.py", label="Home", icon="🏠", disabled=True)
st.page_link("pages/classify.py", label="Classify", icon="1️⃣")
st.page_link("pages/dataset.py", label="Dataset", icon="2️⃣")

st.write("# Ecole Polytechnique de Thies")
st.write("## Génie Informatique et Télécommunications")
st.write("### Classe : DIC3")
st.write("#### Cours : MLOPS")
st.write("PROJET : Classifiez des biens de consommation")
st.write("Présenté par : Ass NIANG & Abdou SAKHO")
