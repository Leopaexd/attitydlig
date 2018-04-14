# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Klassificering


from keras.models import Sequential
from keras.layers import *
from keras import regularizers
import numpy as np
import time
from hyperas.distributions import uniform, choice
from hyperopt import STATUS_OK

def create_model(x_training, y_training, x_testing, y_testing, embedding_matrix):
    start = time.time()
    model = Sequential()
    model.add(Embedding(np.shape(embedding_matrix)[0],
                             output_dim=300,
                             weights=[embedding_matrix],
                             input_length=300,
                             trainable=True))
    model.add(Conv1D(filters={{choice([64,100,200])}}, kernel_size=3,
                          padding='same', activation='relu', kernel_regularizer={{choice([none,regularizers.l2(0.01)])}}))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters={{choice([64,100,200])}}, kernel_size=4,
                          padding='same', activation='relu', kernel_regularizer={{choice([none,regularizers.l2(0.01)])}}))
    model.add(MaxPooling1D(pool_size={{choice([2,3])}}))
    model.add(Conv1D(filters={{choice([64,100,200])}}, kernel_size=5,
                          padding='same', activation='relu', kernel_regularizer={{choice([none,regularizers.l2(0.01)])}}))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout({{uniform(0,0.6)}}))
    model.add(Flatten())
    model.add(Dense(units=250, activation={{choice(['relu','sigmoid'])}}))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adadelta',
                       metrics=['binary_accuracy'])

    model.fit(x_training, y_training, epochs=5, batch_size={{choice([16, 32, 64, 128])}}, verbose=1, validation_split=0.1)
    time_elapsed = time.time() - start
    print("Model fit in ", ("%.2f" % time_elapsed), "seconds")
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = score[1]
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}