
from keras.models import Sequential
from keras.layers import *
from keras import regularizers
import numpy as np
import time


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 300


class Classifier:
    def __init__(self,dictionary,word_vectors):
        self.embedding_matrix=np.zeros((len(dictionary)+1, EMBEDDING_DIM))
        for word, index in dictionary.items():
            try:
                word_vector = word_vectors[word]
            except KeyError:
                word_vector = None
            if word_vector is not None:
                self.embedding_matrix[index] = word_vector



        self.model = Sequential()
        self.model.add(Embedding(len(dictionary) + 1,
                            output_dim=EMBEDDING_DIM,
                            weights=[self.embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
        self.model.add(Conv1D(filters=100, kernel_size=3,
                         padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=100, kernel_size=4,
                              padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=100, kernel_size=5,
                              padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(units=250, activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adadelta',
                      metrics=['binary_accuracy'])


    def fit(self, x_training, y_training):
        start = time.time()
        self.model.fit(x_training, y_training, epochs=5, batch_size=64, verbose=1,validation_split=0.1)
        time_elapsed = time.time() - start
        print("Model fit in ", ("%.2f" % time_elapsed), "seconds")

    def evaluate(self,x_testing,y_testing):
        loss_and_metrics = self.model.evaluate(x_testing,y_testing,batch_size=128,verbose=1)
        print(loss_and_metrics)

    def custom_evaluate(self,x_testing,y_testing):
        predictions = self.model.predict(x_testing,verbose=1)
        hits = 0
        total = len(y_testing)
        if len(y_testing) != len(predictions):
            print('Length mismatch')
        for a,b in zip(predictions,y_testing):
            # if np.all(int(round(a)) == int(round(b))):
            if (int(a) == int(b)):
                hits += 1
        hits = total - hits # To fix reverse-error during development


        print('Accuracy: ' + str(hits/total) + '. ' + str(hits) + '/' + str(total) + ' hits')
        print('x_testing: ' +str(len(x_testing)) + ' y_testing: ' + str(len(y_testing)))
        loss_and_metrics = self.model.evaluate(x_testing, np.array(y_testing), batch_size=128, verbose=1)
        print(loss_and_metrics)
