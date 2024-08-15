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

from modelo_city import recommend_city

# Título del encabezado
st.title("Sistema de Recomendación")

# Selección del modelo de recomendación
model_choice = st.radio(
    "Selecciona el modelo de recomendación:",
    ("Modelo 1", "Modelo 2")
)

# Mostrar el modelo seleccionado
st.write(f"Has seleccionado: {model_choice}")

# Lógica para cada modelo
if model_choice == "Modelo 1":
    recommender = Recommender()
    data = recommender.data

    categorias = data['category'].str.split(', ').explode().unique()
    categorias = list(categorias.tolist())

    categorias_set = set()

    for categoria in categorias:
        # Limpiar los caracteres no deseados
        categoria = categoria.replace("'", "").replace("[", "").replace("]", "")
        
        # Agregar la categoría limpia al conjunto
        categorias_set.add(categoria)

    categoria_deseada = st.selectbox(
        'Selecionar Categoría', categorias_set)

    estado_deseado = st.selectbox('Selecionar Estado', data['state'].unique())

    ciudad_deseado = st.selectbox(
        'Selecionar Ciudad', data[data["state"] == estado_deseado]["city"].unique())

    if st.button('Recomendar Restaurantes'):
        if "restaurant" in categoria_deseada:
            categoria_deseada = categoria_deseada.replace("restaurant", "")
        
        st.write(recommender.recomendar_restaurantes(
            categoria_deseada, estado_deseado, ciudad_deseado))

    
    
elif model_choice == "Modelo 2":
    
    data = pd.read_json('data/dataML.json', lines=True)

    states = data['state'].unique()

    state = st.selectbox('Selecionar Estado', states)

    categories = data[data['state'] == state]['category'].explode().unique().tolist()

    category = st.selectbox('Selecionar Categoria', categories)

    if st.button('Recomendar'):
        try:
            st.write(recommend_city(state, category))
        except Exception as e:
            st.write('No hay suficientes datos de esta categoria en el estado elegido')