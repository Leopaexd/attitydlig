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
import review_translator

def main():
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\olive\\Desktop\\'
                                                                   'Datasets_for_thesis\\'
                                                                   'Swedish Word Vectors\\'
                                                                   'swectors-300dim.txt', binary=True,
                                                                   unicode_errors='ignore')

    directory = 'C:\\Users\\olive\\Desktop\\Datasets_for_thesis\\Prisjakt\\training_data'

    extracted_data = extractor.json_extract(directory)
    extracted_reviews = extracted_data[0]
    polarities = extracted_data[1]

    preprocessed_reviews = preprocessor.preprocess(extracted_reviews)


    # review_translator.translate_reviews(preprocessed_reviews, polarities)

    dictionary = dict.Dictionary(preprocessed_reviews).dictionary

    vectorized_data = vectorizer.vectorize_data(preprocessed_reviews,dictionary,300) # 300 word maximum length
    x_training = vectorized_data[0]
    x_testing = vectorized_data[1]

    # split into training and testing data
    y_training_and_testing = np.array_split(np.asarray(polarities),[0,int(len(polarities)*0.9)])
    y_training =  y_training_and_testing[1]
    y_testing = y_training_and_testing[2]

    pos = 0
    tot = 0
    for value in y_training:
        tot += 1
        if value == 1:
            pos += 1
    print('training data pos: ' + str(pos) + '/' + str(tot) +' = ' + str(pos/tot))

    pos = 0
    tot = 0
    for value in y_testing:
        tot += 1
        if value == 1:
            pos += 1
    print('validation split pos: ' + str(pos) + '/' + str(tot) + ' = ' + str(pos / tot))


    # Translated data
    with open('untranslated_reviews validation combined.txt','r') as file:
        untranslated_reviews = np.concatenate(vectorizer.vectorize_data(preprocessor.preprocess(file.readlines()),dictionary,300))
    with open ('translated_polarities validation combined.txt','r') as file:
        translated_polarities = []
        for line in file:
            translated_polarities.append(int(line))
    pos = 0
    tot = 0
    for value in translated_polarities:
        tot += 1
        if value == 1:
            pos += 1
    print('testing data pos: ' + str(pos) + '/' + str(tot) + ' = ' + str(pos / tot))




    NN_classifier = classifier.Classifier(dictionary,word_vectors)
    NN_classifier.fit(x_training, y_training)
    # NN_classifier.evaluate(x_testing, y_testing)
    # NN_classifier.evaluate(untranslated_reviews, np.array(translated_polarities))
    NN_classifier.custom_evaluate(untranslated_reviews, translated_polarities)


main()
