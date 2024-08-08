from pathlib import Path
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
import math
import os
import joblib

from modelo import Recommender

st.set_page_config(

)

recommender = Recommender()
data = recommender.data

categoria_deseada = st.dropdown(
    'Selecionar Categoría', data['category'].unique())

estado_deseado = st.selectbox('Selecionar Estado', data['state'].unique())

ciudad_deseado = st.selectbox(
    'Selecionar Ciudad', data[data["state"] == estado_deseado]["city"].unique())

if st.button('Recomendar Restaurantes', on_click=recommender.recomendar_restaurantes(categoria_deseada, estado_deseado, ciudad_deseado)):
    st.write(recommender.recomendar_restaurantes(
        categoria_deseada, estado_deseado, ciudad_deseado))
