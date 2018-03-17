# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Vektorisering

import numpy as np
import time

# Return the vector representation of a review, using a specified dictionary.
# First version - simple vector representation including word order but having a maximum length.
def vectorize_review(review,dictionary,length):
    vector = []
    for word in review:
        if len(vector) < length:
            vector.append(dictionary.dictionary[word])
    # Add padding until vector is of specified length
    while len(vector) < length:
        vector.append(0)
    array = np.asarray(vector)
    print(array.shape)
    return array

# Vectorize a list of reviews, and return a numpy 2D matrix of the stacked vectors
def vectorize_data(preprocessed_data,dictionary, length):
    start = time.time()
    print('Vectorizing...')
    vectorized_data = []
    for review in preprocessed_data:
        vectorized_data.append(vectorize_review(review,dictionary, length))
    time_elapsed = time.time() - start
    print("Vectorization completed in ", ("%.2f" % time_elapsed), "seconds")

    x_training_and_testing = np.split(vectorized_data, len(vectorized_data) * 0.9)

    x_training = np.vstack(x_training_and_testing[0])
    x_testing = np.vstack(x_training_and_testing[1])
    return [x_training,x_testing]
