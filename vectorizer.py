# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Vektorisering

import numpy as np

# Return the vector representation of a review, using a specified dictionary.
# First version - simple vector representation including word order but having a maximum length.
def vectorize_review(review,dictionary,length):
    vector = []
    for word in review:
        vector.append(dictionary.dictionary[word])
    # Add padding until vector is of specified length
    while len(vector) < length:
        vector.append(0)
    return np.asarray(vector)

# Vectorize a list of reviews, and return a numpy 2D matrix of the stacked vectors
def vectorize_data(preprocessed_data,dictionary, length):
    vectorized_data = []
    for review in preprocessed_data:
        vectorized_data.append(vectorize_review(review,dictionary, length))
    return np.asarray(vectorized_data)
