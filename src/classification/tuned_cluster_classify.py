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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Dropout, Input, MaxPool3D, GRU, Reshape, TimeDistributed, LSTM, GlobalMaxPool2D, MaxPool2D, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal

from keras.models import Model
from keras.optimizers import adam
from tensorflow.keras.optimizers import Adam
import jsonLog as JL
import importlib


##############################################################
# Parse Args & non-edit Vars
##############################################################

model_name = sys.argv[2] 
subdirectory = "models"
model_name = model_name.strip(".py")
module_name = f"{subdirectory}.{model_name}"
module = importlib.import_module(module_name)
testSubjects = json.loads(sys.argv[1])
dataset = sys.argv[3]
clusterset = sys.argv[4]
jsonDir = f"../logs/clustering_logs/{clusterset}/"
print(jsonDir)

#tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set GPU memory growth option
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to allocate only as much GPU memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Print error message if setting GPU memory growth fails
        print(e)


##############################################################
# Human Vars
##############################################################

batch_size = 200
run_note = "tuned_70_30_DS8_kurtosis_30_properClusters_model5D_coarse_selected" #XXX remember to turn on normalization if not using batch

##############################################################
# Data Sources
##############################################################

csv_label_1 = f"../data/datasets/{dataset}/sequences/MI_RLH_T1_annotation.csv"
npy_label_1 = f"../data/datasets/{dataset}/sequences/MI_RLH_T1.npy"

csv_label_2 = f"../data/datasets/{dataset}/sequences/MI_FF_T2_annotation.csv"
npy_label_2 = f"../data/datasets/{dataset}/sequences/MI_FF_T2.npy"

training_files = [npy_label_1, npy_label_2]

csv_label_1_testing = f"../data/datasets/{dataset}/sequences/MI_RLH_T1_annotation.csv"
npy_label_1_testing = f"../data/datasets/{dataset}/sequences/MI_RLH_T1.npy"

csv_label_2_testing = f"../data/datasets/{dataset}/sequences/MI_FF_T2_annotation.csv"
npy_label_2_testing = f"../data/datasets/{dataset}/sequences/MI_FF_T2.npy"

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

def data_for_subject(npy_file, indices):
    npySubSet = []
    npyLoad = np.load(npy_file)
    maxAxis = (npyLoad.shape[0])-1
    for x in indices:
        if x <= maxAxis: 
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

def get_clustered_indices(class_number, json_filename):
    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)
    class_key = f"class_{class_number}"
    if class_key not in json_data:
        return []  # Return an empty list if the class number is not found
    class_info = json_data[class_key]
    training_indices = class_info.get("trainingIndices", [])
    return training_indices

def get_testing_indices(class_number, json_filename):
    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)
    class_key = f"class_{class_number}"
    if class_key not in json_data:
        return []  # Return an empty list if the class number is not found
    class_info = json_data[class_key]
    testing_indices = class_info.get("testingIndices", [])
    return testing_indices

def create_data(csv_label, subjects, npy_label):
    indices_label = get_indices_for_subject(csv_label, subjects)
    npyData_label = np.array(data_for_subject(npy_label, indices_label))
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


# XXX turned off normalization to try batch normalization XXX 
def prepareDataOld(data):
    # Reshape the data to 2D
    data_2d = data.reshape((-1, data.shape[-1]))
    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_2d)
    # Reshape the data back to 4D
    data_normalized = data_normalized.reshape(data.shape)
    return data_normalized

def prepareData(data):
    return data
    # NORMALIZE Sub-Epochs
    data_min = np.min(data, axis=(1, 2, 3), keepdims=True)
    data_max = np.max(data, axis=(1, 2, 3), keepdims=True)
    epsilon = np.finfo(float).eps
    data = (data - data_min) / (data_max - data_min + epsilon)
    return data

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def classify(training_data_array, tune_data_array, testing_data_array):

    # list of params
    numLabels = len(training_data_array)

    print("################################################################") ##NOTE should make this extensible for diff num classes
    print(f"training data 1 is of length: {len(training_data_array[0])}")
    print(f"training data 2 is of length: {len(training_data_array[1])}")
    print(f"tune data 1 is of length: {len(tune_data_array[0])}")
    print(f"tune data 2 is of length: {len(tune_data_array[1])}")
    print(f"testing data 1 is of length: {len(testing_data_array[0])}")
    print(f"testing data 2 is of length: {len(testing_data_array[1])}")
    print("################################################################")

    # Concatenate training data and labels
    # Train
    train_data = np.concatenate(training_data_array, axis=0)
    train_labels = np.concatenate(make_labels(training_data_array), axis=0)
    # Tune
    tune_data = np.concatenate(tune_data_array, axis=0)
    tune_labels = np.concatenate(make_labels(tune_data_array), axis=0)
    # Test 
    test_data = np.concatenate(testing_data_array, axis=0)
    test_labels = np.concatenate(make_labels(testing_data_array), axis=0)

    # Normalize
    train_data = prepareData(train_data)
    tune_data = prepareData(tune_data)
    test_data = prepareData(test_data)

    # Shuffle the labels and data in unision 
    train_data, train_labels = unison_shuffled_copies(train_data, train_labels)
    tune_data, tune_labels = unison_shuffled_copies(tune_data, tune_labels)
    test_data, test_labels = unison_shuffled_copies(test_data, test_labels)

    ######################################################################
    # Main Model 
    ######################################################################

    model = module.model(numLabels) ### in ./models
    model.summary()
    config = model.to_json()
    JL.model_log(config)

    ######################################################################
    # Optimize
    ######################################################################

    learning_rate = 0.00023274
    optimizer = Adam(learning_rate=learning_rate)

    ######################################################################
    # Call backs 
    ######################################################################

    filepath = "best_model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    dropout_callback = DynamicDropoutCallback(threshold=0.1, high_dropout=0.8, low_dropout=0.4)
    json_logger = JL.JSONLogger('epoch_performance.json')
    earlystop = EarlyStopping(monitor='val_accuracy', patience=5,verbose=1, mode='max')

    callbacks_list = [checkpoint, earlystop, json_logger]

    ######################################################################
    # Compile and fit model : training
    ######################################################################

    global batch_size
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=50, batch_size=batch_size,validation_data=(test_data, test_labels),callbacks=callbacks_list)

    ######################################################################
    # Compile and fit model : tuning 
    ######################################################################

    optimizer = Adam(learning_rate=learning_rate)
    earlystop = EarlyStopping(monitor='val_accuracy', patience=10,verbose=1, mode='max')

    # load the weights that yielded the best validation accuracy
    model.load_weights('best_model.hdf5')
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(tune_data, tune_labels, epochs=25, batch_size=50,validation_data=(test_data, test_labels),callbacks=callbacks_list)

    ######################################################################
    # Evaluate the best model
    ######################################################################

    model.load_weights("best_model.hdf5")
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    print('Test Loss:', test_loss)
    print('Test Accuracy: %.3f' % test_accuracy)

    global accuracy
    global loss
    accuracy = test_accuracy
    loss = test_loss


def get_testing_indices_Old(class_number, json_filename):
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

    print("###############################################################")
    print(f"running subject: {testingSubjects[0]}") 
    print("###############################################################")
    subjects = []

    #npy_label_1  
    #class1 = 3 : MI_RLH_T1.npy
    class1 = 3
    #class2 = 4 : MI_RLH_T2.npy
    class2 = 2

    jsonPath = jsonDir + get_subject_file(jsonDir, testingSubjects[0])
    ###############################################################
    # for clustering 
    ###############################################################
    data_1 = (np.array(data_for_subject(npy_label_1, get_similar_indices(class1, jsonPath))))
    data_2 = (np.array(data_for_subject(npy_label_2, get_similar_indices(class2, jsonPath))))
    #np.random.shuffle(data_1) 
    #np.random.shuffle(data_2) 

    ###############################################################
    # for whole subject 
    ###############################################################
    #testingSubjects = list(map(int, testingSubjects))
    #excluded_numbers = [88, 92, 100, 104, testingSubjects[0]]
    #subjects = [num for num in range(1, 110) if num not in excluded_numbers]
    #data_1 = create_data(csv_label_1, subjects, npy_label_1)
    #data_2 = create_data(csv_label_2, subjects, npy_label_2)

    ###############################################################
    # tuning for subject (currently based on clustered) 
    ###############################################################
    tuning_data_1 = (np.array(data_for_subject(npy_label_1, get_clustered_indices(class1, jsonPath))))
    tuning_data_2 = (np.array(data_for_subject(npy_label_2, get_clustered_indices(class2, jsonPath))))

    ###############################################################
    # testing for subject (currently based on clustered) 
    ###############################################################
    test_data_1 = (np.array(data_for_subject(npy_label_1, get_testing_indices(class1, jsonPath))))
    test_data_2 = (np.array(data_for_subject(npy_label_2, get_testing_indices(class2, jsonPath))))

    # Shuffle all data
    np.random.shuffle(test_data_1)
    np.random.shuffle(test_data_2)
    np.random.shuffle(data_1) 
    np.random.shuffle(data_2) 
    np.random.shuffle(tuning_data_1) 
    np.random.shuffle(tuning_data_2) 

    # make arrays for each set of data
    test_data = []
    test_data.append(test_data_1)
    test_data.append(test_data_2)

    data = []
    data.append(data_1)
    data.append(data_2)

    tune_data = []
    tune_data.append(tuning_data_1)
    tune_data.append(tuning_data_2)
    print(f"tuning data 1 shape is {tuning_data_1.shape}")
    print(f"tuning data 2 shape is {tuning_data_2.shape}")

    global batch_size
    classify(data, tune_data, test_data)
    global run_note
    JL.output_log(subjects, testingSubjects, training_files, testing_files, accuracy, run_note, dataset, batch_size)
    JL.make_logs()


runSubject(testSubjects)
