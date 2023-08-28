import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Dropout, Input, MaxPool3D, GRU, Reshape, TimeDistributed, LSTM, GlobalMaxPool2D, MaxPool2D, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal

from keras.models import Model
from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam

def model(numLabels):

    input_layer = Input((80, 17, 17, 1))
    conv1 = Conv3D(filters=8, kernel_size=(7, 3, 3), activation='relu')(input_layer)
    dropout1 = Dropout(0.3)(conv1)  # Add dropout after Conv3D
    conv2 = Conv3D(filters=16, kernel_size=(5, 3, 3), activation='relu')(dropout1)
    conv3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv2)
    flatten_layer = Flatten()(conv3)
    reshape_layer = Reshape((-1, flatten_layer.shape[1]))(flatten_layer)

    # GRU layers to capture time-dependent information
    gru_layer1 = GRU(units=80, return_sequences=True)(reshape_layer)
    dropout2 = Dropout(0.5)(gru_layer1)  # Add dropout after GRU
    gru_layer2 = GRU(units=80)(dropout2)

    # Fully connected layer
    dense_layer = Dense(units=256, activation='relu')(gru_layer2)

    # Output layer
    output_layer = Dense(units=numLabels, activation='softmax')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
