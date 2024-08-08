import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import time


class Recommender:
    def __init__(self) -> None:
        self.data = joblib.load('data/data.pkl')
        self.vectorizador = TfidfVectorizer()

        output_file = 'data/similitud_restaurantes.pkl'
        input_dir = 'data/'
        num_parts = len([name for name in os.listdir(input_dir)
                        if name.startswith('similitud_restaurantes.pkl.part')])

        print(self.data[self.data['category'].str.contains("mexican restaurant", case=False)])

        self.recomponer_archivo(output_file, input_dir, num_parts)
        self.similitud = joblib.load('data/similitud_restaurantes.pkl')
        self.tfidf_matrix = self.vectorizador.fit_transform(self.data['text'])

    def recomponer_archivo(self, output_file, input_dir, num_parts):
        with open(output_file, 'wb') as f:
            for i in range(num_parts):
                part_file = os.path.join(
                    input_dir, f'similitud_restaurantes.pkl.part{i}')
                
                with open(part_file, 'rb') as chunk_file:
                    f.write(chunk_file.read())

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
        
        print(recomendaciones[["name", "address", "rating"]].head(2))
        
        return recomendaciones[["name", "address", "rating"]]
