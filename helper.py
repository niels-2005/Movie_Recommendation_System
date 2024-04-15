import pandas as pd 
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json


# TODO: Add Docstrings!


def extract_genre_names(genre_string):
    genre_list = json.loads(genre_string)
    return ', '.join([genre['name'] for genre in genre_list])


def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


def create_summary(row):
    return f"Title: {row['original_title']}\n\n" \
           f"Overview: {row['overview']}\n\n" \
           f"Genres: {row['genre_names']}\n\n" \
           f"Release: {row['release_date']}\n\n" \
           f"Runtime: {row['runtime']} min\n\n" \
           f"Voting Average: {row['vote_average']}"


def get_dataframe():
    path = "./data/tmdb_5000_prep.csv"
    df = pd.read_csv(path)
    return df


def get_cosine_sim_matrix_and_indices(df: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words='english')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    #Construct a reverse map of indices and movie titles
    indices = pd.Series(df.index, index=df['original_title'])
    return cosine_sim, indices



def get_recommendations(title: str, n_recomm: int = 10, only_title: bool = True):
    df = get_dataframe()
    cosine_sim, indices = get_cosine_sim_matrix_and_indices(df=df) 
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:n_recomm+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    if only_title:
        return df["original_title"].iloc[movie_indices]
    else:
        return df["summary"].iloc[movie_indices].tolist()


