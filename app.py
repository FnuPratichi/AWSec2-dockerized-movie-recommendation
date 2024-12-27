from flask import Flask, request, render_template, jsonify
import pandas as pd
from src.recommender import get_recommendations
from src.preprocessing import preprocess_data
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

# Preprocess data and load the necessary models once when the app starts
preprocessed_data, tfidf_vectors = preprocess_data()
print(preprocessed_data.head())
print(tfidf_vectors)

@app.route('/')
def home():
    return render_template('home.html')  # Assuming you have a home.html template for movie title input

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        movie_title = request.form.get('title')
    else:
        movie_title = request.args.get('title')

    if not movie_title:
        return jsonify({'error': 'Please provide a movie title'}), 400

    # Get recommendations for the provided movie title
    recommendations = get_recommendations(movie_title, preprocessed_data, tfidf_vectors)

    if recommendations.empty:
        return jsonify({'error': 'No recommendations available'}), 404

    # Format recommendations as a list of dictionaries for easy rendering in the frontend
    recommendations_list = recommendations.to_dict(orient='records')

    return jsonify(recommendations_list)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

