#build the NN models: RNN module
import tensorflow
from tensorflow.python.ops import control_flow_ops
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import LSTM, Embedding,SimpleRNN
from keras.utils import np_utils
from tensorflow.python.platform import flags
from numpy import shape
import numpy as np
from skimage import io, color, exposure, transform
import os
import glob
import h5py
import pandas as pd
import numpy


FLAGS = flags.FLAGS
#tensorflow.python.control_flow_ops =control_flow_ops


def rnn_model(seq_length, input_dim):
    model = Sequential()
    model.add((SimpleRNN(64, input_shape=(seq_length, input_dim))))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(1,init='normal'))
    return model


