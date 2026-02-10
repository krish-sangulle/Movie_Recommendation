from flask import Flask, render_template ,request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
movies_df = pd.read_csv('dataset/bollywood_movies.csv')

#Combine relevant features into a single string
def combine_features(row):
    return f"{row['Genre']} {row['Director']}  {row['Actors']}  {row['Plot']}"

movies_df['combined_features'] = movies_df.apply(combine_features, axis=1)

vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(feature_matrix)

# Movie recommendation logic
def recommend_movies(movie_title,genre_filter=None):
    if movie_title not in movies_df['Title'].values:
        return[]
    
    #Otherwise, find the index location of given inside dataframe
    index = movies_df[movies_df['Title'] == movie_title].index[0]
    #Use cosine similarity algo to find other similar movies as a list
    similar_movies = list(enumerate(cosine_sim[index]))
    #Sort Movie Names based on Cosine Similarity Score
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:10]
    recommendations = []
    for i in sorted_movies:
        movie = movies_df.iloc[i[0]]
        #If generation is selected not match with the current movie(ignore it)
        if genre_filter and genre_filter.lower() not in movie['Genre'].lower():
            continue
        recommendations.append(movie)
        if len(recommendations) == 5:
            break
    return recommendations

@app.route('/')
def index():
    movie_titles = list(movies_df['Title'].values)
    #Pick all genertions (Set - Ignore duplicate (unique))
    genres = set(g for movie in movies_df['Genre'].str.split('|') for g in movie)
    return render_template('index.html', movie_titles = movie_titles, genres = sorted(genres))

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie']
    genres_filter = request.form.get('genre')
    recommendations = recommend_movies(movie_title,genres_filter)
    result = [{
        'title':movie['Title'],         'poster':movie['Poster'],
        'genre':movie['Genre'],         'director':movie['Director'],
        'actors':movie['Actors']
    } for movie in recommendations ]
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)