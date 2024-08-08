import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
class Recommender:
    def __init__(self) -> None:
        self.data = joblib('data/data.pkl')
        self.vectorizador = TfidfVectorizer()
        
        output_file = 'similitud_recompuesto.pkl' 
        input_dir = 'data/'  
        num_parts = len([name for name in os.listdir(input_dir) if name.startswith('similitud.pkl.part')])
        
        self.recomponer_archivo(output_file, input_dir, num_parts)
        self.similitud = joblib('data/similitud_restaurantes.pkl')
        
    def recomponer_archivo(self, output_file, input_dir, num_parts):
        with open(output_file, 'wb') as f:
            for i in range(num_parts):
                part_file = os.path.join(input_dir, f'similitud.pkl.part{i}')
                with open(part_file, 'rb') as chunk_file:
                    f.write(chunk_file.read())

    
    def recomendar_restaurantes(self, categoria, estado, ciudad):
        filtrado = self.data[(self.data['state'] == estado.upper()) & (self.data['category'].str.contains(categoria, case=False)) & (self.data['city'] == ciudad.lower())]

        if filtrado.empty:
            return pd.DataFrame()  # No hay restaurantes que coincidan

        filtrado_matrix = self.vectorizador.transform(filtrado['text'])
        similitud_filtrado = cosine_similarity(filtrado_matrix)


        promedio_similitud = similitud_filtrado.mean(axis=1)

        filtrado['similitud_promedio'] = promedio_similitud

        recomendaciones = filtrado.sort_values(by='similitud_promedio', ascending=False)
        return recomendaciones[["name", "address", "rating"]]