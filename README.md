# Movie_Recommendation_System

Welcome to the **Movie_Recommendation_System** repository! This project utilizes machine learning techniques to recommend movies based on their similarity in content. It is built using Python, pandas, numpy, and scikit-learn to process movie data and generate recommendations.

## Project Overview

The **Movie_Recommendation_System** employs a content-based filtering approach using the TF-IDF vectorizer and cosine similarity to recommend movies. This system is designed to help users discover new movies similar to their favorites.

## Features

- **Content-Based Filtering**: Uses textual content from movie descriptions to recommend similar movies.
- **TF-IDF Vectorization**: Converts text data into a matrix of TF-IDF features to measure textual relevance.
- **Cosine Similarity**: Computes similarity scores based on the cosine angle between vectors, representing movie descriptions.
- **Customizable Recommendations**: Users can specify the number of recommendations and whether to receive detailed summaries or just titles.

## Methodology

1. **Data Loading**: Movie data is loaded into pandas DataFrames from CSV files.
2. **Preprocessing**: Genres are extracted from JSON strings and text data is cleaned and prepared.
3. **TF-IDF Vectorization**: Movie descriptions are transformed into TF-IDF vectors.
4. **Similarity Calculation**: A cosine similarity matrix is generated from the TF-IDF vectors.
5. **Recommendation Generation**: Based on the similarity scores, movies are recommended to the user.

This system focuses on providing recommendations that are not only based on genres but also on detailed descriptions, offering a nuanced selection tailored to the user's preferences.
