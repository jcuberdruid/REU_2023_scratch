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
# Same as model4 but with batch normalization and changed weights and kernal sizes 
################################################################

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1


def model(numLabels):
    model = Sequential()
    model.add(Input(shape=(64, 640, 3)))  # Shape of input images
    model.add(Conv2D(filters=8, kernel_size=(13, 5), activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(9, 4), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(5, 3), activation='relu'))
    model.add(Flatten())
    # Reshape to have 3D data before GRU, (batch_size, timesteps, features)
    # You will have to determine the appropriate values for timesteps and features
    model.add(Reshape((20, -1)))  # These are placeholder values, adjust based on your needs
    model.add(GRU(units=80, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(GRU(units=80))
    model.add(Dense(units=512, activation='relu', kernel_regularizer=l1(0.003), name='dense'))
    model.add(Dropout(rate=0.2, name="dropout_2"))
    model.add(Dense(units=256, activation='relu', kernel_regularizer=l1(0.003), name='dense_1'))
    model.add(Dense(units=numLabels, activation='softmax', name='dense_3'))

    return model
