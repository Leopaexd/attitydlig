# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Huvudfil

import new_classifier as classifier
import dictionary as dict
import extractor
import preprocessor
import vectorizer
import numpy as np
import gensim
from keras.utils.np_utils import to_categorical

def main():
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\olive\\Desktop\\'
                                                                   'Datamängder för uppsats\\'
                                                                   'Swedish Word Vectors\\'
                                                                   'swectors-300dim.txt', binary=True,
                                                                   unicode_errors='ignore')

    directory = 'C:\\Users\\olive\\Desktop\\Datamängder för uppsats\\Prisjakt' #\\Utvärderingsdata'

    extracted_data = extractor.json_extract(directory)
    extracted_reviews = extracted_data[0]
    polarities = extracted_data[1]

    preprocessed_reviews = preprocessor.preprocess(extracted_reviews)
    dictionary = dict.Dictionary(preprocessed_reviews).dictionary

    vectorized_data = vectorizer.vectorize_data(preprocessed_reviews,dictionary,300) # 300 word maximum length
    x_training = vectorized_data[0]
    x_testing = vectorized_data[1]


    # split into training and testing data
    y_training_and_testing = np.array_split(np.asarray(polarities),[0,int(len(polarities)*0.9)])
    y_training =  y_training_and_testing[1]
    y_testing = y_training_and_testing[2]
    NN_classifier = classifier.Classifier(dictionary,word_vectors)
    NN_classifier.fit(x_training, y_training)
    NN_classifier.evaluate(x_testing, y_testing)
    NN_classifier.custom_evaluate(x_testing, y_testing)
    print(y_training)

    ones = 0
    minus = 0
    for x in y_testing:
        if x == 1:
            ones += 1
        else:
            minus+=1
    print (str(ones) +' ones and ' +str(minus) + ' minus ones')

main()
