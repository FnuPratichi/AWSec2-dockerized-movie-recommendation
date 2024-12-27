import os
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import re
import logging
import pickle


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings for cleaner output
import warnings
warnings.simplefilter('ignore')

# Function to normalize text (remove spaces, special characters, and lowercase)
def normalize_text(text):
    return re.sub(r'\W+', '', text.lower())

# Function to preprocess and prepare data
def preprocess_data():
    # Fetch file paths from environment variables
    movies_path = os.getenv('MOVIES_PATH', 'data/movies_metadata.csv')
    ratings_path = os.getenv('RATINGS_PATH', 'data/ratings.csv')
    keywords_path = os.getenv('KEYWORDS_PATH', 'data/keywords.csv')
    credits_path = os.getenv('CREDITS_PATH', 'data/credits.csv')

    # Load data
    movies = pd.read_csv(movies_path, low_memory=False)
    ratings = pd.read_csv(ratings_path)
    keywords = pd.read_csv(keywords_path)
    credits = pd.read_csv(credits_path)

    # Data Cleaning
    # Ensure 'crew' column is parsed correctly as a list of dictionaries
    movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    movies['production_companies'] = movies['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    movies['production_countries'] = movies['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    movies['spoken_languages'] = movies['spoken_languages'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # Ensure 'crew' and 'cast' are parsed correctly as lists of dictionaries
    credits['crew'] = credits['crew'].apply(literal_eval)
    credits['cast'] = credits['cast'].apply(literal_eval)

    # Filter movies based on vote count
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.90)
    
    movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

    df = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())][['id','title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
    df['w_ratings'] = (df['vote_count'] / (df['vote_count'] + m)) * df['vote_average'] + (m / (df['vote_count'] + m)) * C

    # Merge credits and keywords data
    credits = credits.drop_duplicates(subset='id', keep='first').reset_index(drop=True)
    keywords = keywords.drop_duplicates(subset='id', keep='first').reset_index(drop=True)
    df['id'] = df['id'].astype('int')
    df2 = df.merge(credits, on='id').merge(keywords, on='id')

    # Ensure 'id' is of the expected dtype: np.int64
    df2['id'] = df2['id'].astype(np.int64)

    # Normalize text and create tf-idf vectors
    df2['director'] = df2['crew'].apply(lambda x: next((i['name'] for i in x if i['job'] == 'Director'), None))

    # Update main_cast to extract names from dictionaries (previous issue)
    df2['main_cast'] = df2['cast'].apply(lambda x: [actor['name'] for actor in x[:4]] if isinstance(x, list) else [])

    # Handle any NaN or None values in 'main_cast' before applying TF-IDF
    df2['main_cast'] = df2['main_cast'].apply(lambda x: x if isinstance(x, list) else [])
    df2['director'] = df2['director'].fillna('')

    df2['normalized_title'] = df2['title'].apply(normalize_text)

    # TF-IDF Vectorization for genres, cast, and director
    tfidf_genres = TfidfVectorizer(stop_words='english')
    genres_tfidf = tfidf_genres.fit_transform(df2['genres'].apply(lambda x: ' '.join(x)))

    tfidf_cast = TfidfVectorizer(stop_words='english')
    cast_tfidf = tfidf_cast.fit_transform(df2['main_cast'].apply(lambda x: ' '.join(x)))

    tfidf_director = TfidfVectorizer(stop_words='english')
    director_tfidf = tfidf_director.fit_transform(df2['director'])

    # Combine all tf-idf features
    combined_tfidf = hstack([genres_tfidf, cast_tfidf, director_tfidf])

    # Validate the DataFrame after processing
    #validate_columns(df2)
    logging.info("Starting preprocessing")
    df.to_csv('processed_data.csv', index=False)
    with open('combined_tfidf.pkl', 'wb') as f:
        pickle.dump(combined_tfidf, f)
    logging.info("Preprocessing completed")

    return df2, combined_tfidf
