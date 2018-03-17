# Author: Oliver Glant - oliver.glant@gmail.com
# Attitydlig - attitydanalys på svenska
# Klassificering


from keras.models import Sequential
from keras.layers import Dense


class Classifier:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu',input_dim=300)
        self.model.add(Dense(units=10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def fit(self, x_training, y_training):
        self.model.fit(x_training, y_training, epochs=5, batch_size=32, verbose=3)

    def evaluate(self,x_testing,y_testing):
        loss_and_metrics = self.model.evaluate(x_testing,y_testing,batch_size=128,verbose=1)
        print(loss_and_metrics)
