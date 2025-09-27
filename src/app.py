import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os

# Configuracion de la pagina
st.set_page_config(
    page_title=" Recomendador de Películas",
    page_icon="🎬",
    layout="wide"
)

# Estilo
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #E31836;
    }
    </style>
""", unsafe_allow_html=True)

# Ttulo principal
st.markdown("""
    # Título principal
st.title("Sistema de Recomendación de Películas")
""", unsafe_allow_html=True)

# bryan 
@st.cache_resource
def load_pipeline(model_path: str):
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

#Maria
@st.cache_data
def load_data(     ):
    
    



    

