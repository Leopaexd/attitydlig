# Model based on http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1105494&dswid=-9724

from keras.models import Sequential
from keras.layers import *
import numpy as np
import time


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 300
TRAINABLE = True



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
                            trainable=False))

        self.model.add(Conv1D(filters=100, kernel_size=4,
                         padding='same', activation='relu'))

        self.model.add(MaxPooling1D(pool_size=2))



        self.model.add(Flatten())

        self.model.add(Dense(units=250, activation='relu'))

        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adadelta',
                      metrics=['binary_accuracy','binary_accuracy'])


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
            if np.all(int(a) == int(b)):
                hits += 1
                print(str(a) + ' = ' + str(b))
            else: print (str(a) + ' != ' + str(b))


        print('Accuracy: ' + str(hits/total) + '. ' + str(hits) + '/' + str(total) + ' hits')
        print('x_testing: ' +str(len(x_testing)) + ' y_testing: ' + str(len(y_testing)))
        loss_and_metrics = self.model.evaluate(x_testing, y_testing, batch_size=128, verbose=1)
        print(loss_and_metrics)



"""
# Instantiate a sequential model
self.model = Sequential()
# Embedding layer as the first layer

model.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=3,
            batch_size=32)
# Evaluate model on new data...
scores = model.evaluate(X_test, y_test)
# ...or generate predictions based on new data
predictions = model.predict(X_test)
"""