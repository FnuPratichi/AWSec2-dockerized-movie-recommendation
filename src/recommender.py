# -*- coding: utf-8 -*-
"""recommender.py"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load preprocessed data and TF-IDF vectors
def load_data():
    try:
        df = pd.read_csv('processed_movies_data.csv')
        tfidf_vectors = joblib.load('combined_tfidf.pkl')
        logging.info("Data and TF-IDF vectors loaded successfully.")
        return df, tfidf_vectors
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Normalize text (remove spaces, special characters, and lowercase)
def normalize_text(text):
    return re.sub(r'\W+', '', str(text).strip().lower())

# Generate cosine similarity matrix
def compute_cosine_similarity(tfidf_vectors):
    logging.info("Computing cosine similarity matrix...")
    return cosine_similarity(tfidf_vectors, tfidf_vectors)

# Get recommendations based on cosine similarity
def get_recommendations(title, df, cosine_sim, k=5):
    normalized_input = normalize_text(title)
    logging.info(f"Normalized Input Title: {normalized_input}")
    
    # Ensure the dataset has the required column
    if 'normalized_title' not in df.columns:
        logging.error("Dataset does not contain 'normalized_title'.")
        return pd.DataFrame()
    
    try:
        # Find the index of the movie title
        idx = df[df['normalized_title'] == normalized_input].index[0]
        logging.info(f"Movie found at index: {idx}")
        cosine_sim = cosine_similarity(cosine_sim, cosine_sim)
        # Compute similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:k+1]  # Skip the first one (self-match)
        print(sim_scores)
        # Get movie indices and fetch details
        movie_indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[movie_indices][['title', 'year', 'genres', 'w_ratings','director', 'main_cast']]
        return recommendations
    except IndexError:
        logging.error(f"Title '{title}' not found in the dataset.")
        return pd.DataFrame(columns=['title', 'year', 'genres', 'w_ratings','director', 'main_cast'])

# Main function
if __name__ == "__main__":
    # Load data
    try:
        df, tfidf_vectors = load_data()
    except Exception as e:
        logging.critical("Critical error during data loading. Exiting.")
        exit()

    # Normalize titles in the dataset
    if 'normalized_title' not in df.columns:
        logging.info("Adding 'normalized_title' to dataset...")
        df['normalized_title'] = df['title'].apply(normalize_text)
    
    # Compute cosine similarity matrix
    try:
        cosine_sim = compute_cosine_similarity(tfidf_vectors)
    except Exception as e:
        logging.error(f"Error computing cosine similarity: {e}")
        exit()

    # Example usage
    title = input("Enter a movie title for recommendations: ")
    recommendations = get_recommendations(title, df, cosine_sim, k=10)

    if not recommendations.empty:
        print("\nRecommended Movies:\n")
        print(recommendations.to_string(index=False))
    else:
        print("\nNo recommendations available for the given title.")
