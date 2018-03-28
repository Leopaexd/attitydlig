# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Vektorisering

import numpy as np
import time
import gensim

# Return the vector representation of a review, using a specified dictionary.
# First version - simple vector representation including word order but having a maximum length.
def vectorize_review(review,dictionary,length):
    vector = []
    for word in review:
        try:
            if len(vector) < length:
                vector.append(dictionary[word])
        # Ignore words not in dictionary/vector model
        except KeyError:
            pass
    # Add padding until vector is of specified length
    while len(vector) < length:
        vector.append(0)
    array = np.asarray(vector)
    return array

# Vectorize a list of reviews, and return a numpy 2D matrix of the stacked vectors
def vectorize_data(preprocessed_data,dictionary, length):
    start = time.time()
    print('Vectorizing...')
    vectorized_data = []
    for review in preprocessed_data:
        vectorized_data.append(vectorize_review(review,dictionary, length))
    vectorized_data = np.vstack(vectorized_data)
    time_elapsed = time.time() - start
    print("Vectorization completed in ", ("%.2f" % time_elapsed), "seconds")

    x_training_and_testing = np.array_split(vectorized_data, [0,int((len(vectorized_data) * 0.9))])
    print(len(x_training_and_testing))

   # x_training = np.vstack(x_training_and_testing[0])
   # x_testing = np.vstack(x_training_and_testing[1])

    x_training = x_training_and_testing[0] # 2st
    x_testing = x_training_and_testing[1] # 2 st


    return [x_training,x_testing]