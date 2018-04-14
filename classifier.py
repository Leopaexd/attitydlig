# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Klassificering


from keras.models import Sequential
from keras.layers import *
from keras import regularizers
import numpy as np
import time

def create_model(x_training, y_training, x_testing, y_testing):
    start = time.time()
    model = Sequential()
    model.add(Embedding(len(dictionary) + 1,
                             output_dim=EMBEDDING_DIM,
                             weights=[self.embedding_matrix],
                             input_length=MAX_SEQUENCE_LENGTH,
                             trainable=True))
    model.add(Conv1D(filters=100, kernel_size=3,
                          padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=100, kernel_size=4,
                          padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=100, kernel_size=5,
                          padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=250, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta',
                       metrics=['binary_accuracy'])

    model.fit(x_training, y_training, epochs=5, batch_size=64, verbose=1, validation_split=0.1)
    time_elapsed = time.time() - start
    print("Model fit in ", ("%.2f" % time_elapsed), "seconds")
    return model