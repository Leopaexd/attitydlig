# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Vektorisering

import numpy as np

# Return the vector representation of a review, using a specified dictionary.
def vectorize_review(review,dictionary):
    return vector

# Vectorize a list of reviews, and return a numpy 2D matrix of the stacked vectors
def vectorize_data(preprocessed_data,dictionary):
    vectorized_data = []
    for review in preprocessed_data:
        vectorized_data.append(vectorize_review(review,dictionary))
    return np.asarray(vectorized_data)
