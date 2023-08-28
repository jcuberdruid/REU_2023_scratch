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

def model(numLabels):
    input_layer = Input((80, 17, 17, 1))

    # First stack
    conv1 = Conv3D(filters=8, kernel_size=(13, 5, 4), activation='relu')(input_layer)
    dropout1 = Dropout(0.3)(conv1)
    batchnorm1 = BatchNormalization()(dropout1)
    conv2 = Conv3D(filters=16, kernel_size=(9, 4, 4), activation='relu')(batchnorm1)
    conv3 = Conv3D(filters=32, kernel_size=(5, 3, 3), activation='relu')(conv2)
    flatten_layer = Flatten()(conv3)
    reshape_layer = Reshape((-1, flatten_layer.shape[1]))(flatten_layer)
    gru_layer1 = GRU(units=80, return_sequences=True)(reshape_layer)
    dropout2 = Dropout(0.5)(gru_layer1)
    batchnorm2 = BatchNormalization()(dropout2)
    gru_layer2 = GRU(units=80)(batchnorm2)
    dense_layer = Dense(units=512, activation='relu', kernel_regularizer=l1(0.005))(gru_layer2)

    # Second stack
    conv4 = Conv3D(filters=8, kernel_size=(11, 4, 3), activation='relu')(input_layer)
    dropout3 = Dropout(0.3)(conv4)
    batchnorm3 = BatchNormalization()(dropout3)
    conv5 = Conv3D(filters=16, kernel_size=(7, 3, 3), activation='relu')(batchnorm3)
    conv6 = Conv3D(filters=32, kernel_size=(3, 2, 2), activation='relu')(conv5)
    flatten_layer2 = Flatten()(conv6)
    reshape_layer2 = Reshape((-1, flatten_layer2.shape[1]))(flatten_layer2)
    gru_layer3 = GRU(units=80, return_sequences=True)(reshape_layer2)
    dropout4 = Dropout(0.5)(gru_layer3)
    batchnorm4 = BatchNormalization()(dropout4)
    gru_layer4 = GRU(units=80)(batchnorm4)
    dense_layer2 = Dense(units=512, activation='relu', kernel_regularizer=l1(0.005))(gru_layer4)

    # Merge the two stacks
    merged = tf.keras.layers.concatenate([dense_layer, dense_layer2])

    # Output layer
    output_layer = Dense(units=numLabels, activation='softmax')(merged)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

