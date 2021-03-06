# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Huvudfil

import dictionary as dict
import extractor
import preprocessor
import vectorizer
import numpy as np
import gensim
import review_translator
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from keras.models import Sequential
from keras.layers import *
from keras import regularizers
from keras import backend
import time
from hyperas.distributions import uniform, choice
from hyperopt import STATUS_OK
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import  f1_score, precision_score, recall_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (' — val_f1: % f — val_precision: % f — val_recall % f' % (_val_f1, _val_precision, _val_recall))
        return

metrics = Metrics()

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def prepare_data():
    directory = 'C:\\Users\\olive\\Desktop\\Datasets_for_thesis\\Prisjakt\\training_data'
    extracted_data = extractor.json_extract(directory)
    extracted_reviews = extracted_data[0]
    polarities = extracted_data[1]
    preprocessed_reviews = preprocessor.preprocess(extracted_reviews)
    dictionary = dict.Dictionary(preprocessed_reviews).dictionary
    vectorized_data = vectorizer.vectorize_data(preprocessed_reviews, dictionary, 300)  # 300 word maximum length
    x_train = vectorized_data[0]
    x_test = vectorized_data[1]

    # split into training and testing data
    y_training_and_testing = np.array_split(np.asarray(polarities), [0, int(len(polarities) * 0.9)])
    y_train = y_training_and_testing[1]
    y_test = y_training_and_testing[2]

    # creation of embedding matrix
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\olive\\Desktop\\'
                                                                   'Datasets_for_thesis\\'
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

    return (x_train, y_train, x_test, y_test, create_embedding_matrix)

def translated_data():
    directory = 'C:\\Users\\olive\\Desktop\\Datasets_for_thesis\\Prisjakt\\training_data'
    extracted_data = extractor.json_extract(directory)
    extracted_reviews = extracted_data[0]
    polarities = extracted_data[1]
    preprocessed_reviews = preprocessor.preprocess(extracted_reviews)
    dictionary = dict.Dictionary(preprocessed_reviews).dictionary
    # review_translator.translate_reviews(preprocessed_reviews, polarities)
    with open('untranslated_reviews validation combined.txt', 'r') as file:
        untranslated_reviews = np.concatenate(
            vectorizer.vectorize_data(preprocessor.preprocess(file.readlines()), dictionary, 300))
    with open('translated_polarities validation combined.txt', 'r') as file:
        translated_polarities = []
        for line in file:
            translated_polarities.append(int(line))
    return [untranslated_reviews,np.array(translated_polarities)]

def create_model(x_train, y_train, x_test, y_test, embedding_matrix):
    def matthews_correlation(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = K.sum(y_pos * y_pred_pos)
        tn = K.sum(y_neg * y_pred_neg)

        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)

        numerator = (tp * tn - fp * fn)
        denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / (denominator + K.epsilon())

    backend.clear_session() # prevent excessive memory use
    start = time.time()
    model = Sequential()
    model.add(Embedding(np.shape(embedding_matrix)[0],
                             output_dim=300,
                             weights=[embedding_matrix],
                             input_length=300,
                             trainable=True))
    model.add(Conv1D(filters=100, kernel_size=3,
                          padding='same', activation='relu', kernel_regularizer=None))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=100, kernel_size=4,
                          padding='same', activation='relu', kernel_regularizer=None))
    model.add(MaxPooling1D(pool_size={{choice([2,3])}}))
    model.add(Conv1D(filters=100, kernel_size=5,
                          padding='same', activation='relu', kernel_regularizer={{choice([None,regularizers.l2(0.01)])}}))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(units=250, activation={{choice(['relu','sigmoid'])}}))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer={{choice(['adadelta','adam'])}},
                       metrics=[matthews_correlation,'binary_accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1, validation_split=0.1)
    time_elapsed = time.time() - start
    print("Model fit in ", ("%.2f" % time_elapsed), "seconds")
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = score[1]
    return {'loss': -accuracy, 'status': STATUS_OK} #, 'model': model

def main():

    # data = prepare_data(preprocessed_reviews, polarities, dictionary)
    test_data = translated_data()

    best_run, best_model = optim.minimize(model=create_model,
                              data=prepare_data,
                              algo=tpe.suggest,
                              max_evals=10, #
                              trials=Trials())
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(test_data[0],test_data[1]))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    with open('best_parameters.txt', 'w') as output_file:
        for parameter in best_run:
            output_file.write(parameter + ': '+ str(best_run[parameter])  + '\n')



    # NN_classifier = classifier.create_model(data[0],data[1],data[2],data[3],data[4])
    # print(NN_classifier.evaluate(test_data[0],test_data[1]))

main()
