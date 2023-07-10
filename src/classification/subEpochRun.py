import multiprocessing
import sys
import os
import json
from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import datetime
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Dropout, Input, MaxPool3D, GRU, Reshape, TimeDistributed, LSTM, GlobalMaxPool2D, MaxPool2D, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal

from keras.models import Model
from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam
import jsonLog as JL

accuracy = None
loss = None

dataset = "processed4"

# generalization: specific output classes
# output expirment data
csv_label_1 = f"../data/datasets/{dataset}/sequences/MI_RLH_T1_annotation.csv"
npy_label_1 = f"../data/datasets/{dataset}/sequences/MI_RLH_T1.npy"

csv_label_2 = f"../data/datasets/{dataset}/sequences/MI_RLH_T2_annotation.csv"
npy_label_2 = f"../data/datasets/{dataset}/sequences/MI_RLH_T2.npy"

training_files = [npy_label_1, npy_label_2]

csv_label_1_testing = f"../data/datasets/{dataset}/sequences/MI_RLH_T1_annotation.csv"
npy_label_1_testing = f"../data/datasets/{dataset}/sequences/MI_RLH_T1.npy"

csv_label_2_testing = f"../data/datasets/{dataset}/sequences/MI_RLH_T2_annotation.csv"
npy_label_2_testing = f"../data/datasets/{dataset}/sequences/MI_RLH_T2.npy"

testing_files = [npy_label_1_testing, npy_label_2_testing]

'''
csv_label_3 = "../data/datasets/sequences/MI_FF_T1_annotation.csv"
npy_label_3 = "../data/datasets/sequences/MI_FF_T1.npy"

csv_label_4 = "../data/datasets/sequences/MI_FF_T2_annotation.csv"
npy_label_4 = "../data/datasets/sequences/MI_FF_T2.npy"
'''

class DynamicDropoutCallback(Callback): #XXX is meh
    def __init__(self, threshold=0.1, high_dropout=0.8, low_dropout=0.4):
        super(DynamicDropoutCallback, self).__init__()
        self.threshold = threshold
        self.high_dropout = high_dropout
        self.low_dropout = low_dropout

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        epoch_accuracy = logs.get('accuracy')
        epoch_val_accuracy = logs.get('val_accuracy')

        if epoch_accuracy is not None and epoch_val_accuracy is not None:
            delta = abs(epoch_accuracy - epoch_val_accuracy)

            if delta > self.threshold:
                current_dropout = self.high_dropout
            elif delta < -self.threshold and epoch_val_accuracy > epoch_accuracy:
                current_dropout = 0.0
            else:
                current_dropout = self.low_dropout

            for layer in self.model.layers:
                if isinstance(layer, Dropout):
                    layer.rate = current_dropout

def generate_random_numbers(length, trainingPercent):
    subjects = [x for x in range(1, 110) if x not in [88, 92, 100, 104]]
    random.shuffle(subjects)
    testingSubjects = subjects.copy()
    numTestingSubjects = int(length*trainingPercent)
    while (len(testingSubjects) != numTestingSubjects):
        testingSubjects.pop(0)
    subjects = subjects[: len(subjects) - (105-length)]
    return subjects, testingSubjects


def get_indices_for_subject(csv_file, subjects):
    indices = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for subject in subjects:
                # Convert subject to integer
                if int(row['subject']) == subject and int(row['index']) % 9 != 0:
                    indices.append(int(row['index']))
    #random.shuffle(indices)
    print(len(indices))
    return indices

#XXX so just need to write a script to open and read the indices from the json files
def data_for_subject(npy_file, indices):
    npySubSet = []
    npyLoad = np.load(npy_file)
    for x in indices:
        npySubSet.append(npyLoad[x])
    return npySubSet

def get_similar_indices(class_number, json_filename):
    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)

    class_key = f"class_{class_number}"
    if class_key not in json_data:
        return []  # Return an empty list if the class number is not found

    class_info = json_data[class_key]
    similar_indices = class_info.get("similarIndices", [])

    return similar_indices


def create_data(csv_label, subjects, npy_label):
    indices_label = get_indices_for_subject(csv_label, subjects)
    npyData_label = np.array(data_for_subject(npy_label, indices_label))
    # np.save('label.npy', npyData_label)
    # return np.load('label.npy')
    return npyData_label

###############################################################
###############################################################


def make_labels(data_array):
    label_array = []
    # [np.zeros(data_label_1.shape[0]), np.ones(data_label_2.shape[0])]
    # np.full(shape, fill_value[, dtype, order, like])
    for index, x in enumerate(data_array):
        label_array.append(np.full(x.shape[0], index))
    return label_array


def prepareData(data):
    # Reshape the data to 2D
    data_2d = data.reshape((-1, data.shape[-1]))
    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_2d)
    # Reshape the data back to 4D
    data_normalized = data_normalized.reshape(data.shape)
    return data_normalized


def classify(training_data_array, testing_data_array):
    # list of params
    numLabels = len(training_data_array)
    print("################################################################")
    print(len(training_data_array[0]))
    print(len(testing_data_array[0]))
    print("################################################################")
    # Load label_1 and label_2 data from .npy files
    # data_label_1 = np.load('label_1.npy')
    # data_label_2 = np.load('label_2.npy')

    # Concatenate training data and labels
    training_data = np.concatenate(training_data_array, axis=0)
    training_labels = np.concatenate(make_labels(training_data_array), axis=0)

    # Concatenate testing  data and labels
    testing_data = np.concatenate(testing_data_array, axis=0)
    testing_labels = np.concatenate(make_labels(testing_data_array), axis=0)

    training_data_normalized = prepareData(training_data)
    testing_data_normalized = prepareData(testing_data)

    #np.random.shuffle(training_data_normalized)
    #np.random.shuffle(testing_data_normalized)
    print(training_data_normalized.shape)
    print(testing_data_normalized.shape)

    # Split the data into training and testing sets
    # train_data, test_data, train_labels, test_labels = train_test_split(data_normalized, labels, test_size=0.1, random_state=42)

    train_data = training_data_normalized
    train_labels = training_labels
    test_data = testing_data_normalized
    test_labels = testing_labels

    ######################################################################
    # Main Model -- TODO implement logging (may have to save entire section)
    ######################################################################
    # Reshape the data to match the input shape for the 3D-CNN
    # Define the model

    input_layer = Input((80, 17, 17, 1))
    model = keras.Sequential()    

    # Input layer
    model.add(Input(shape=(80, 17, 17, 1), dtype="float32", name="input_layer"))

    # Conv3D layer
    model.add(Conv3D(filters=32, kernel_size=(5, 1, 1), strides=(1, 1, 1), padding='valid', data_format='channels_last', activation='relu', name='conv3d'))

    # Dropout layer
    model.add(Dropout(rate=0.3, name="dropout"))

    # Flatten layer
    model.add(Flatten(data_format="channels_last", name="flatten"))

    # Reshape layer
    model.add(Reshape(target_shape=(-1, 702848), name="reshape"))

    # First GRU layer
    model.add(GRU(units=80, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name='gru'))

    # Second Dropout layer
    model.add(Dropout(rate=0.5, name="dropout_1"))

    # Second GRU layer
    model.add(GRU(units=80, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', name='gru_1'))

    # Dense layers
    model.add(Dense(units=512, activation='relu', name='dense'))
    model.add(Dropout(rate=0.5, name="dropout_2"))
    model.add(Dense(units=256, activation='relu', name='dense_1'))
    model.add(Dense(units=2, activation='softmax', name='dense_2'))

    model.summary()
    config = model.to_json()
    JL.model_log(config)
   ######################################################################
    ######################################################################
    # Create the optimizer with the desired learning rate
    learning_rate = 1e-4
    optimizer = Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dropout_callback = DynamicDropoutCallback(threshold=0.1, high_dropout=0.8, low_dropout=0.4)
    json_logger = JL.JSONLogger('epoch_performance.json')
    # Compile the model
    model.fit(train_data, train_labels, epochs=50, batch_size=100,
              validation_data=(test_data, test_labels), callbacks=[
                  json_logger,
                  keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
               #   dropout_callback
                ])
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)
    global accuracy
    global loss
    accuracy = test_accuracy
    loss = test_loss


def get_testing_indices(class_number, json_filename):
    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)

    class_key = f"class_{class_number}"
    if class_key not in json_data:
        return []  # Return an empty list if the class number is not found

    class_info = json_data[class_key]
    testing_indices = class_info.get("testingIndices", [])

    return testing_indices

def get_subject_file(path, subject_number):
    # convert subject_number to string and prepend "S" to it
    subject_string = 'S' + str(subject_number) + '_'
    
    # list all files in the directory
    files = os.listdir(path)
    
    # iterate over all files
    for file in files:
        # check if file starts with subject_string
        if file.startswith(subject_string):
            # if so, return file name
            return file
            
    # if no file is found for the subject, return None
    return None

def runSubject(testingSubjects):

    #print(get_similar_indices(1, "S1_clustering_log_2023_07_06_10_25_04"))
    #data_for_subject(npy_file, indices):
    #npyData_label = np.array(data_for_subject(npy_label, indices_label))

    #exclude = testingSubjects + [88, 89, 92, 100, 104] 
    #subjects = [x for x in range(1, 20) if x not in exclude]
    #random.shuffle(subjects)
    print("###############################################################")
    print(f"running subject: {testingSubjects[0]}") 
    print("###############################################################")
    subjects = []
    npy_label_1  
    #class1 = 3 : MI_RLH_T1.npy
    class1 = 3
    #class2 = 4 : MI_RLH_T2.npy
    class2 = 4
    #S41_clustering_log_2023_07_06_10_36_39 
    #json Path 
    #jsonPath = '../clustering_logs/processed/S51_clustering_log_2023_07_06_10_39_21'
    jsonPath = '../clustering_logs/processed4/' + get_subject_file('../clustering_logs/processed4/', testingSubjects[0])
    print(jsonPath)
    data_1 = (np.array(data_for_subject(npy_label_1, get_similar_indices(class1, jsonPath))))
    data_2 = (np.array(data_for_subject(npy_label_2, get_similar_indices(class2, jsonPath))))
    np.random.shuffle(data_1) 
    np.random.shuffle(data_2) 

    #data_1 = create_data(csv_label_1, subjects, npy_label_1)
    #data_2 = create_data(csv_label_2, subjects, npy_label_2)

    print(f"data_1 is {data_1.shape}")

    test_data_1 = (np.array(data_for_subject(npy_label_1, get_testing_indices(class1, jsonPath))))
    test_data_2 = (np.array(data_for_subject(npy_label_2, get_testing_indices(class2, jsonPath))))
    np.random.shuffle(test_data_1)
    np.random.shuffle(test_data_2)

    #test_data_1 = create_data(csv_label_1_testing, testingSubjects, npy_label_1_testing)
    #test_data_2 = create_data(csv_label_2_testing, testingSubjects, npy_label_2_testing)
    
    test_data = []
    test_data.append(test_data_1)
    test_data.append(test_data_2)

    data = []
    data.append(data_1)
    data.append(data_2)

    classify(data, test_data)
    JL.output_log(subjects, testingSubjects,training_files, testing_files, accuracy)
    JL.make_logs()


runSubject([int(sys.argv[1])])
