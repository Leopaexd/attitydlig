# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Huvudfil

import classifier
import dictionary as dict
import extractor
import preprocessor
import vectorizer
import gensim
import numpy as np

def main():
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\olive\\Desktop\\'
                                                                   'Datamängder för uppsats\\'
                                                                   'Swedish Word Vectors\\'
                                                                   'swectors-300dim.txt', binary=True,
                                                                   unicode_errors='ignore')

    directory = 'C:\\Users\\olive\\Desktop\\Datamängder för uppsats\\Prisjakt\\Utvärderingsdata'

    extracted_data = extractor.json_extract(directory)
    extracted_reviews = extracted_data[0]
    polarities = extracted_data[1]

    #for line in extracted_reviews:
       # print(line)

    preprocessed_reviews = preprocessor.preprocess(extracted_reviews)
    #dictionary = dict.Dictionary(preprocessed_reviews)
    vectorized_data = vectorizer.vectorize_data(preprocessed_reviews, word_vectors, 300)  # 300 word maximum length
    x_training = vectorized_data[0]
    x_testing = vectorized_data[1]
    # split into training and testing data
    y_training_and_testing = np.array_split(np.asarray(polarities), [0, int(len(polarities) * 0.9)])
    y_training = y_training_and_testing[1]
    y_testing = y_training_and_testing[2]

    tpot = classifier.Classifier()
    tpot.fit(x_training,y_training)

main()
