#build the NN models: RNN module
import tensorflow

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

def dnn_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(4,init='normal', activation='softmax'))
    return model