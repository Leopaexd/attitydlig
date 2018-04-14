# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Huvudfil

import classifier as classifier
import dictionary as dict
import extractor
import preprocessor
import vectorizer
import numpy as np
import gensim
import review_translator
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe


def create_embedding_matrix(dictionary):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\olive\\Desktop\\'
                                                                   'Datamängder för uppsats\\'
                                                                   'Swedish Word Vectors\\'
                                                                   'swectors-300dim.txt', binary=True,
                                                                   unicode_errors='ignore')
    embedding_matrix = np.zeros((len(dictionary) + 1, 300))
    for word, index in dictionary.items():
        try:
            word_vector = word_vectors[word]
        except KeyError:
            word_vector = None
        if word_vector is not None:
            embedding_matrix[index] = word_vector
    return embedding_matrix

def prepare_data(preprocessed_reviews, polarities, dictionary):
    vectorized_data = vectorizer.vectorize_data(preprocessed_reviews, dictionary, 300)  # 300 word maximum length
    x_training = vectorized_data[0]
    x_testing = vectorized_data[1]

    # split into training and testing data
    y_training_and_testing = np.array_split(np.asarray(polarities), [0, int(len(polarities) * 0.9)])
    y_training = y_training_and_testing[1]
    y_testing = y_training_and_testing[2]

    return (x_training, y_training, x_testing, y_testing, create_embedding_matrix(dictionary))

def translated_data(dictionary):
    # review_translator.translate_reviews(preprocessed_reviews, polarities)
    with open('untranslated_reviews validation combined.txt', 'r') as file:
        untranslated_reviews = np.concatenate(
            vectorizer.vectorize_data(preprocessor.preprocess(file.readlines()), dictionary, 300))
    with open('translated_polarities validation combined.txt', 'r') as file:
        translated_polarities = []
        for line in file:
            translated_polarities.append(int(line))
    return [untranslated_reviews,np.array(translated_polarities)]

def main():
    directory = 'C:\\Users\\olive\\Desktop\\Datamängder för uppsats\\Prisjakt\\Utvärderingsdata'
    extracted_data = extractor.json_extract(directory)
    extracted_reviews = extracted_data[0]
    polarities = extracted_data[1]
    preprocessed_reviews = preprocessor.preprocess(extracted_reviews)
    dictionary = dict.Dictionary(preprocessed_reviews).dictionary

    data = prepare_data(preprocessed_reviews, polarities, dictionary)
    test_data = translated_data(dictionary)

    best_run, best_model = optim.minimize(model=classifier.create_model,
                              data=data,
                              algo=tpe.suggest,
                              max_evals=10,
                              trials=Trials())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(test_data[0],test_data[1]))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    # NN_classifier = classifier.create_model(data[0],data[1],data[2],data[3],data[4])
    # print(NN_classifier.evaluate(test_data[0],test_data[1]))



main()
