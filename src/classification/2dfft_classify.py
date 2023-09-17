import multiprocessing
from tensorflow.keras.preprocessing import image as kimage
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
from sklearn.preprocessing import LabelEncoder


##############################################################
# Parse Args & non-edit Vars
##############################################################

subdirectory = "models"
model_name = sys.argv[2] 
model_name = model_name.strip(".py")
module_name = f"{subdirectory}.{model_name}"
module = importlib.import_module(module_name)
testSubjects = json.loads(sys.argv[1])
dataset = sys.argv[3]
clusterset = sys.argv[4]
#class_files = sys.argv[5] #TODO currently not set to add this argument in the ../main.py 

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
run_note = "test run delete later"#XXX remember to turn on normalization if not using batch

##############################################################
# Data Sources
##############################################################
#pass array of files to include via "class_files" list (see above) 
#

#training_files = []
'''
for x in class_files:
    training_files.append(f"../data/datasets/{dataset}/sequences/MI_RLH_T1.npy")
'''
#testing_files = testing_files

def apply_cutout_on_dataset(dataset, cutout_size=2, p=0.25):
    # Dataset shape: (n, 80, 17, 17)
    # Cutout applied to: (80, 17, 17)

    def apply_cutout_on_sample(sample):
        # Sample shape: (80, 17, 17)
        mask_value = sample.min()

        h, w = sample.shape[1], sample.shape[2]
        cutout_h = np.random.randint(cutout_size//2, cutout_size)
        cutout_w = np.random.randint(cutout_size//2, cutout_size)

        for _ in range(80):  # Apply cutout on each (17, 17) feature map
            if np.random.rand() < p:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - cutout_h // 2, 0, h)
                y2 = np.clip(y + cutout_h // 2, 0, h)
                x1 = np.clip(x - cutout_w // 2, 0, w)
                x2 = np.clip(x + cutout_w // 2, 0, w)

                sample[:, y1:y2, x1:x2] = mask_value

        return sample

    # Apply the function on the dataset
    dataset = np.array([apply_cutout_on_sample(sample) for sample in dataset])

    return dataset



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

###############################################################
###############################################################

def unison_shuffled_copies(a, b):
    assert a.shape[0] == b.shape[0], "First dimension must be the same size for both arrays."
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

def classify(train_data, train_labels, tune_data, tune_labels, test_data, test_labels):

    # list of params
    numLabels = len(set(train_labels)) 

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

    learning_rate = 9.8747e-05
    optimizer = Adam(learning_rate=learning_rate)

    ######################################################################
    # Call backs 
    ######################################################################

    filepath = "best_model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    #dropout_callback = DynamicDropoutCallback(threshold=0.1, high_dropout=0.8, low_dropout=0.4)
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
    model.fit(tune_data, tune_labels, epochs=50, batch_size=50,validation_data=(test_data, test_labels),callbacks=callbacks_list)

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

def load_images_and_labels(file_paths):
    all_images = []
    all_labels = []

    for full_path in file_paths:
        if full_path.endswith('.png'):
            img = kimage.load_img(full_path, target_size=(64, 640))
            img_array = kimage.img_to_array(img)

            label = os.path.basename(os.path.dirname(full_path))

            all_images.append(img_array)
            all_labels.append(label)
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    return all_images, all_labels

def separate_files_by_subject(directory, target_subject):
    target_files = []
    other_files = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith('.png'):
                subject_identifier = fname.split('_')[0]
                full_path = os.path.join(root, fname)
                if subject_identifier == target_subject:
                    target_files.append(full_path)
                else:
                    other_files.append(full_path)
    return target_files, other_files


def runSubject(testingSubjects, source_directory):

    print("###############################################################")
    print(f"running subject: {testingSubjects[0]}") 
    print("###############################################################")

    # Separate files by target subject
    target_files, other_files = separate_files_by_subject(source_directory, 'S57')
    # Load images and labels for target subject
    target_images, target_labels = load_images_and_labels(target_files)
    # Load images and labels for other subjects
    train_data, train_labels = load_images_and_labels(other_files)


    label_encoder = LabelEncoder()

    target_labels = label_encoder.fit_transform(target_labels)
    train_labels = label_encoder.fit_transform(train_labels)

    split_index = target_images.shape[0] // 2
    tune_data = target_images[:split_index]
    tune_labels = target_labels[:split_index]

    test_data = target_images[split_index:]
    test_labels = target_labels[split_index:]

    global batch_size
    classify(train_data, train_labels, tune_data, tune_labels, test_data, test_labels)
    global run_note
    JL.output_log(subjects, testingSubjects, training_files, testing_files, accuracy, run_note, dataset, batch_size)
    JL.make_logs()

source_directory = "/home/jason/clone/data/datasets/processed8_79high_MM/binary_test/class_inference"
testSubjects = []
testSubjects.append(57)
runSubject(testSubjects, source_directory)
