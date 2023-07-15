import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Dropout, Input, MaxPool3D, GRU, Reshape, TimeDistributed, LSTM, GlobalMaxPool2D, MaxPool2D, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal

from keras.models import Model
from keras.optimizers import adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
################################################################
# Same as model5A but with L2 regulariation 
################################################################

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

def model(numLabels):
    model = Sequential()
    model.add(Input(shape=(80, 17, 17, 1)))
    model.add(Conv3D(filters=8, kernel_size=(32, 5, 3), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=16, kernel_size=(16, 4, 3), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Conv3D(filters=32, kernel_size=(8, 3, 3), activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Reshape((-1, model.layers[-1].output_shape[1])))
    model.add(GRU(units=80, return_sequences=True))
    model.add(BatchNormalization())
    model.add(GRU(units=80))
    model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.05), name='dense'))
    model.add(Dense(units=numLabels, activation='softmax', name='dense_3'))
    return model

