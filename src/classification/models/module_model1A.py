import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Dropout, Input, MaxPool3D, GRU, Reshape, TimeDistributed, LSTM, GlobalMaxPool2D, MaxPool2D, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal

from keras.models import Model
from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1

def model(numLabels):
    model = keras.Sequential()
    input_layer = Input((80, 17, 17, 1))
    # Input layer
    model.add(Input(shape=(80, 17, 17, 1), dtype="float32", name="input_layer"))
    # Conv3D layer
    model.add(Conv3D(filters=32, kernel_size=(5, 1, 1), strides=(1, 1, 1), padding='valid', data_format='channels_last', activation='relu', name='conv3d2'))
    # Batch normalization
    model.add(BatchNormalization(name="batch_normalization"))
    # Dropout layer
    model.add(Dropout(rate=0.3, name="dropout"))
    # Flatten layer
    model.add(Flatten(data_format="channels_last", name="flatten"))
    # Reshape layer
    model.add(Reshape(target_shape=(-1, 702848), name="reshape"))
    # First GRU layer
    model.add(GRU(units=80, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name='gru'))
    # Second Dropout layer
    model.add(Dropout(rate=0.4, name="dropout_1"))
    # Second GRU layer
    model.add(GRU(units=80, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', name='gru_1'))
    # Dense layers with L1 regularization
    model.add(Dense(units=512, activation='relu', kernel_regularizer=l1(0.002), name='dense'))
    model.add(Dropout(rate=0.4, name="dropout_2"))
    model.add(Dense(units=256, activation='relu', kernel_regularizer=l1(0.002), name='dense_1'))
    model.add(Dense(units=2, activation='softmax', name='dense_2'))
    return model

