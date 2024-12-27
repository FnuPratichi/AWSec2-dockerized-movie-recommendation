import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw_movies.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed_movies.csv')
TFIDF_FILE_PATH = os.path.join(DATA_PATH, 'tfidf_vectors.pkl')