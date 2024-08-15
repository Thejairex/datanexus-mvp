from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import storage
import pandas as pd
import joblib
import gdown
import time
import os


class Recommender:
    def __init__(self) -> None:
        self.data = joblib.load('data/data_restaurantes.pkl')
        self.vectorizador = TfidfVectorizer()

        if not os.path.exists('data/similitud_restaurantes.pkl'):

            # URL del archivo en Google Drive
            url = 'https://drive.google.com/uc?id=1zmF-WBfMPZLR67vqaHs4HuPv9z-iIyoA'

            # Descargar el archivo
            gdown.download(url, 'data/similitud_restaurantes.pkl', quiet=False)
        
        self.similitud = joblib.load('data/similitud_restaurantes.pkl')
        self.tfidf_matrix = self.vectorizador.fit_transform(self.data['text'])

    def download_blob(bucket_name, source_blob_name, destination_file_name):
        # Inicializa el cliente
        storage_client = storage.Client()

        # Accede al bucket
        bucket = storage_client.bucket(bucket_name)

        # Accede al blob (archivo)
        blob = bucket.blob(source_blob_name)

        # Descarga el archivo
        blob.download_to_filename(destination_file_name)

        print(f'Archivo {source_blob_name} descargado a {destination_file_name}.')

    def recomendar_restaurantes(self, categoria, estado, ciudad):
        filtrado = self.data[(self.data['state'] == estado.upper()) & (
            self.data['category'].str.contains(categoria, case=False)) & (self.data['city'] == ciudad)]

        if filtrado.empty:
            return pd.DataFrame()  # No hay restaurantes que coincidan

        filtrado_matrix = self.vectorizador.transform(filtrado['text'])
        similitud_filtrado = cosine_similarity(filtrado_matrix)

        promedio_similitud = similitud_filtrado.mean(axis=1)

        filtrado['similitud_promedio'] = promedio_similitud

        recomendaciones = filtrado.sort_values(
            by='similitud_promedio', ascending=False)
                
        return recomendaciones[["name", "address", "rating"]]
