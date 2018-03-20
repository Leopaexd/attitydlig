# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Klassificering


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time

class Classifier:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(units=128, activation='relu',input_dim=300))
        self.model.add(Dense(units=2, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def fit(self, x_training, y_training):
        start = time.time()
        self.model.fit(x_training, y_training, epochs=2000, batch_size=64, verbose=0,validation_split=0.1)
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
            if np.all(a == b): hits += 1
        print('Accuracy: ' + str(hits/total) + '. ' + str(hits) + '/' + str(total) + ' hits')
        print('x_testing: ' +str(len(x_testing)) + ' y_testing: ' + str(len(y_testing)))
        loss_and_metrics = self.model.evaluate(x_testing, y_testing, batch_size=128, verbose=1)
        print(loss_and_metrics)