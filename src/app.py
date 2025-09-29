import streamlit as st

import pandas as pd

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np    



# Configuración de la página
st.set_page_config(
    page_title="Recomendador de Películas",
    page_icon="🎬",
    layout="wide"
)

# Título y descripción
st.title("🎬 Sistema de Recomendación de Películas")
st.markdown("### Encuentra tu próxima película favorita")

# Cargar el modelo y los datos
@st.cache_data
def cargar_datos():
    try:
        # Cargar el pipeline
        with open("models/rotten_pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        
        # Cargar el dataset de películas
        movies = pd.read_csv("data/processed/movies.csv")
        
        return pipeline, movies
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None, None

pipeline, movies = cargar_datos()

if pipeline is not None and movies is not None:
    # Preparar los datos
    movies_clean = movies.copy()
    for col in ["genres", "actors", "directors", "movie_info", "critics_consensus"]:
        movies_clean[col] = movies_clean[col].fillna("")
    movies_clean["release_year"] = movies_clean["release_year"].fillna("1900")
    movies_clean["tomatometer_rating"] = movies_clean["tomatometer_rating"].fillna(0)

    # Sidebar para filtros
    st.sidebar.header("Filtros de Búsqueda")
    
    # Géneros disponibles
    all_genres = []
    for genres in movies_clean["genres"].str.split(","):
        if isinstance(genres, list):
            all_genres.extend(genres)
    unique_genres = sorted(list(set([g.strip() for g in all_genres if g])))
    
    # Filtros en el sidebar
    selected_genres = st.sidebar.multiselect("Géneros", unique_genres)
    
    # Extraer años únicos
    years = sorted(movies_clean["release_year"].unique())
    year_range = st.sidebar.slider(
        "Rango de Años",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )
    
    # Input para actor y director
    selected_actor = st.sidebar.text_input("Actor")
    selected_director = st.sidebar.text_input("Director")
    
    # Slider para Tomatometer
    selected_tomatometer = st.sidebar.slider("Puntuación mínima en Tomatometer", 0, 100, 50)
    
    # Película de referencia
    movie_titles = sorted(movies_clean["movie_title"].unique())
    selected_movie = st.selectbox("Selecciona una película de referencia", [""] + movie_titles)
    
    # Pesos para la recomendación
    st.sidebar.header("Ajustes del Modelo")
    alpha = st.sidebar.slider("Peso de la película seleccionada", 0.0, 1.0, 0.7)
    beta = st.sidebar.slider("Peso de los filtros", 0.0, 1.0, 0.2)
    gamma = st.sidebar.slider("Peso del sentimiento del crítico", 0.0, 1.0, 0.1)
    
    # Número de recomendaciones
    top_n = st.sidebar.number_input("Número de recomendaciones", 1, 20, 5)

    # Función de recomendación
    def recomendar_peliculas():
        # Calcular match_score
        movies_clean["match_score"] = 0
        if selected_genres:
            movies_clean["match_score"] += movies_clean["genres"].apply(
                lambda x: sum(g in x for g in selected_genres)
            )
        if selected_actor:
            movies_clean["match_score"] += movies_clean["actors"].apply(
                lambda x: 2 if selected_actor in x else 0
            )
        if selected_director:
            movies_clean["match_score"] += movies_clean["directors"].apply(
                lambda x: 2 if selected_director in x else 0
            )
        movies_clean["match_score"] += movies_clean["release_year"].apply(
            lambda x: 1 if year_range[0] <= int(str(x)[:4]) <= year_range[1] else 0
        )
        movies_clean["match_score"] += movies_clean["tomatometer_rating"].apply(
            lambda x: 1 if x >= selected_tomatometer else 0
        )

        # Normalizar match_score
        movies_clean["match_norm"] = movies_clean["match_score"] / movies_clean["match_score"].max()

        # Calcular sentimiento
        movies_clean["consensus_sentiment_prob"] = pipeline.predict_proba(
            movies_clean["critics_consensus"]
        )[:, 1]
        movies_clean["consensus_sentiment_norm"] = (
            movies_clean["consensus_sentiment_prob"] / 
            movies_clean["consensus_sentiment_prob"].max()
        )

        # Crear matriz TF-IDF
        movies_clean["combined_features"] = (
            movies_clean["genres"] + " " +
            movies_clean["directors"] + " " +
            movies_clean["actors"] + " " +
            movies_clean["movie_info"]
        )
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(movies_clean["combined_features"])

        if not selected_movie:
            final_score = (
                beta * movies_clean["match_norm"] + 
                gamma * movies_clean["consensus_sentiment_norm"]
            )
            top_indices = final_score.argsort()[::-1][:top_n]
        else:
            idx = movies_clean[movies_clean["movie_title"] == selected_movie].index[0]
            cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            final_score = (
                alpha * cosine_sim +
                beta * movies_clean["match_norm"] +
                gamma * movies_clean["consensus_sentiment_norm"]
            )
            
            top_indices = final_score.argsort()[::-1]
            top_indices = [i for i in top_indices if i != idx]
            top_indices = top_indices[:top_n]

        return movies_clean.iloc[top_indices][
            ["movie_title", "genres", "actors", "directors", "tomatometer_rating", 
             "critics_consensus", "match_score", "consensus_sentiment_prob"]
        ]

    # Botón para generar recomendaciones
    if st.button("Obtener Recomendaciones"):
        with st.spinner("Generando recomendaciones..."):
            recommendations = recomendar_peliculas()
            
            # Mostrar recomendaciones
            for _, movie in recommendations.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Tomatometer", f"{movie['tomatometer_rating']}%")
                        st.metric("Sentimiento", f"{movie['consensus_sentiment_prob']:.2f}")
                    with col2:
                        st.subheader(movie["movie_title"])
                        st.write(f"**Géneros:** {movie['genres']}")
                        st.write(f"**Director:** {movie['directors']}")
                        st.write(f"**Actores principales:** {movie['actors'][:100]}...")
                        if movie['critics_consensus']:
                            st.write(f"**Consenso de críticos:** {movie['critics_consensus']}")
                    st.divider()
else:
    st.error("No se pudieron cargar los datos necesarios para la aplicación.")
    st.info("Por favor, verifica que los archivos de datos y el modelo estén disponibles en las rutas correctas.")