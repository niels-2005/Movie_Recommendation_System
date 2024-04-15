import pandas as pd 
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json


# TODO: Add Docstrings!


def extract_genre_names(genre_string: str) -> str:
    """
    Converts a JSON string of genre information into a comma-separated string of genre names.

    Args:
        genre_string (str): A JSON string containing genre data with 'id' and 'name' keys.

    Returns:
        str: A comma-separated string of genre names.
    """
    genre_list = json.loads(genre_string)
    return ', '.join([genre['name'] for genre in genre_list])


def weighted_rating(x: pd.Series, m: float, C: float) -> float:
    """
    Calculates the weighted rating for each movie using the IMDB formula.

    Args:
        x (pd.Series): A pandas Series containing 'vote_count' and 'vote_average' for the movie.
        m (float): The minimum votes required to be listed in the chart.
        C (float): The mean vote across the whole report.

    Returns:
        float: The weighted rating of the movie.
    """
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


def create_summary(row: pd.Series) -> str:
    """
    Creates a formatted summary string for a movie from its data row.

    Args:
        row (pd.Series): A pandas Series containing movie data.

    Returns:
        str: A formatted string summarizing the movie details.
    """
    return f"Title: {row['original_title']}\n\n" \
           f"Overview: {row['overview']}\n\n" \
           f"Genres: {row['genre_names']}\n\n" \
           f"Release: {row['release_date']}\n\n" \
           f"Runtime: {row['runtime']} min\n\n" \
           f"Voting Average: {row['vote_average']}"


def get_dataframe() -> pd.DataFrame:
    """
    Loads and returns a DataFrame containing movie data from a predefined CSV file path.

    Returns:
        pd.DataFrame: The DataFrame containing the movie data.
    """
    path = "./data/tmdb_5000_prep.csv"
    return pd.read_csv(path)


def get_cosine_sim_matrix_and_indices(df: pd.DataFrame) -> tuple:
    """
    Constructs and returns a cosine similarity matrix and a map of movie titles to their index.

    Args:
        df (pd.DataFrame): A DataFrame containing movie overviews.

    Returns:
        tuple: A tuple containing the cosine similarity matrix and a Series mapping titles to indices.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['original_title']).to_dict()
    return cosine_sim, indices



def get_recommendations(title: str, n_recomm: int = 10, only_title: bool = True) -> list:
    """
    Retrieves a list of recommended movie titles or summaries based on the similarity to a given movie title.

    Args:
        title (str): The title of the movie to find recommendations for.
        n_recomm (int, optional): The number of recommendations to return. Defaults to 10.
        only_title (bool, optional): Whether to return only titles (True) or summaries (False).

    Returns:
        list: A list of movie titles or summaries recommended based on the given movie title.
    """
    df = get_dataframe()
    cosine_sim, indices = get_cosine_sim_matrix_and_indices(df=df) 
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recomm+1]

    movie_indices = [i[0] for i in sim_scores]

    if only_title:
        return df["original_title"].iloc[movie_indices].tolist()
    else:
        return df["summary"].iloc[movie_indices].tolist()


