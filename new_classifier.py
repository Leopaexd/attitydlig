
from keras.models import Sequential
from keras.layers import *
from keras import regularizers
from keras.callbacks import TensorBoard
import numpy as np
import time
from keras.callbacks import Callback
from sklearn.metrics import  f1_score, precision_score, recall_score
from keras import backend as K


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

class Classifier:
    def __init__(self,dictionary,word_vectors):
        self.embedding_matrix=np.zeros((len(dictionary)+1, 300))
        for word, index in dictionary.items():
            try:
                word_vector = word_vectors[word]
            except KeyError:
                word_vector = None
            if word_vector is not None:
                self.embedding_matrix[index] = word_vector



        self.model = Sequential()
        self.model.add(Embedding(len(dictionary) + 1,
                            output_dim=300,
                            weights=[self.embedding_matrix],
                            input_length=300,
                            trainable=True))
        self.model.add(Conv1D(filters=200, kernel_size=3,
                         padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=200, kernel_size=4,
                              padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=200, kernel_size=5,
                              padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.7))
        self.model.add(Flatten())
        self.model.add(Dense(units=250, activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adadelta',
                      metrics=['binary_accuracy', f1,f1,matthews_correlation])


    def fit(self, x_training, y_training):
        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
        start = time.time()
        self.model.fit(x_training, y_training, epochs=5, batch_size=64, verbose=1,validation_split=0.1,
                       callbacks=[tensorboard,metrics])
        time_elapsed = time.time() - start
        print("Model fit in ", ("%.2f" % time_elapsed), "seconds")

    def evaluate(self,x_testing,y_testing):
        loss_and_metrics = self.model.evaluate(x_testing,y_testing,batch_size=128,verbose=1)
        print(loss_and_metrics)
        # print(f1(K.variable(y_testing),K.variable(self.model.predict(x_testing))))

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
