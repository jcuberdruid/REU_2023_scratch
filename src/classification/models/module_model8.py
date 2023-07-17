from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Reshape, BatchNormalization, GRU, transformer
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from transformers import TFAutoModel

def model(numLabels):
    input_layer = Input((80, 17, 17, 1))
    conv1 = Conv3D(filters=8, kernel_size=(13, 5, 4), activation='relu')(input_layer)
    dropout1 = Dropout(0.3)(conv1)
    batchnorm1 = BatchNormalization()(dropout1)  # Add batch normalization after dropout
    conv2 = Conv3D(filters=16, kernel_size=(9, 4, 4), activation='relu')(batchnorm1)
    conv3 = Conv3D(filters=32, kernel_size=(5, 3, 3), activation='relu')(conv2)
    flatten_layer = Flatten()(conv3)

    # Reshape to sequence for Transformer
    reshape_layer = Reshape((-1, flatten_layer.shape[1]))(flatten_layer)

    # Transformer layers to capture dependencies in the sequence
    transformer_layer1 = Transformer(num_heads=8, key_dim=64, ff_dim=256)(reshape_layer)
    dropout2 = Dropout(0.5)(transformer_layer1)
    batchnorm2 = BatchNormalization()(dropout2)  # Add batch normalization after dropout
    transformer_layer2 = Transformer(num_heads=8, key_dim=64, ff_dim=256)(batchnorm2)

    # Fully connected layer with L1 regularization
    dense_layer = Dense(units=512, activation='relu', kernel_regularizer=l1(0.005))(transformer_layer2)

    # Output layer
    output_layer = Dense(units=numLabels, activation='softmax')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
