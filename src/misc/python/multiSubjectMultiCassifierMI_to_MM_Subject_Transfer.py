import numpy as np
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.layers import Conv3D, Flatten, Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import adam

# generalization: specific output classes
# output expirment data

csv_label_1 = "../data/datasets/sequences/MI_RLH_T1_annotation.csv"
npy_label_1 = "../data/datasets/sequences/MI_RLH_T1.npy"

csv_label_2 = "../data/datasets/sequences/MI_RLH_T2_annotation.csv"
npy_label_2 = "../data/datasets/sequences/MI_RLH_T2.npy"


csv_label_1_MM = "../data/datasets/sequences/MM_RLH_T1_annotation.csv"
npy_label_1_MM = "../data/datasets/sequences/MM_RLH_T1.npy"

csv_label_2_MM = "../data/datasets/sequences/MM_RLH_T2_annotation.csv"
npy_label_2_MM = "../data/datasets/sequences/MM_RLH_T2.npy"


'''
csv_label_3 = "../data/datasets/sequences/MI_FF_T1_annotation.csv"
npy_label_3 = "../data/datasets/sequences/MI_FF_T1.npy"

csv_label_4 = "../data/datasets/sequences/MI_FF_T2_annotation.csv"
npy_label_4 = "../data/datasets/sequences/MI_FF_T2.npy"
'''

def generate_random_numbers(length, trainingPercent):
	subjects = [x for x in range(1, 110) if x not in [88, 92, 100, 104]]
	random.shuffle(subjects)
	testingSubjects = subjects.copy()
	numTestingSubjects = int(length*trainingPercent)
	while (len(testingSubjects) != numTestingSubjects):
		testingSubjects.pop(0)
	print(subjects)
	print(len(subjects))
	subjects = subjects[: len(subjects) - (105-length)]
	return subjects, testingSubjects
	

subjects, testingSubjects = generate_random_numbers(20, 0.2)

print(f"number of subjects: {len(subjects)}")
print(subjects)
print(f"number of testingSubjects: {len(testingSubjects)}")
print(testingSubjects)

def get_indices_for_subject(csv_file, subjects):
    indices = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for subject in subjects:
                # Convert subject to integer
                if int(row['subject']) == subject and int(row['index']) % 9 != 0:
                     indices.append(int(row['index']))
    return indices


def data_for_subject(npy_file, indices):
    npySubSet = []
    npyLoad = np.load(npy_file)
    for x in indices:
        npySubSet.append(npyLoad[x])
    return npySubSet


def create_data(csv_label, subjects, npy_label):
	indices_label = get_indices_for_subject(csv_label, subjects)
	npyData_label = np.array(data_for_subject(npy_label, indices_label))
	# np.save('label.npy', npyData_label)
	# return np.load('label.npy')
	return npyData_label


data_1 = create_data(csv_label_1, subjects, npy_label_1)
data_2 = create_data(csv_label_2, subjects, npy_label_2)

test_data_1 = create_data(csv_label_1_MM, testingSubjects, npy_label_1_MM)
test_data_2 = create_data(csv_label_2_MM, testingSubjects, npy_label_2_MM)

test_data = []
test_data.append(test_data_1)
test_data.append(test_data_2)

data = []
data.append(data_1)
data.append(data_2)

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
	return data

def classify(training_data_array, testing_data_array):
	# list of params
	numLabels = len(training_data_array)

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

	np.random.shuffle(training_data_normalized) 
	np.random.shuffle(testing_data_normalized)
	print(training_data_normalized.shape)
	print(testing_data_normalized.shape)

	# Split the data into training and testing sets
	#train_data, test_data, train_labels, test_labels = train_test_split(data_normalized, labels, test_size=0.1, random_state=42)

	train_data = training_data_normalized
	train_labels = training_labels
	test_data = testing_data_normalized
	test_labels = testing_labels

######################################################################
		# Main Model -- TODO implement logging (may have to save entire section)
######################################################################

	# Reshape the data to match the input shape for the 3D-CNN
	train_data = train_data.reshape((-1, 80, 17, 17, 1))
	test_data = test_data.reshape((-1, 80, 17, 17, 1))
	# Define the model
	'''
	model = tf.keras.Sequential([
	    layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu',
	                  input_shape=(80, 17, 17, 1)),
	    layers.MaxPooling3D(pool_size=(2, 2, 2)),
	    layers.Flatten(),
	    layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),  # Dropout layer with a dropout rate of 0.2
	    layers.Dense(numLabels, activation='softmax')
	])
	'''
	input_layer = Input((80, 17, 17, 1))
	conv1 = Conv3D(filters = 8, kernel_size=(7, 3, 3), activation='relu')(input_layer)
	conv2 = Conv3D(filters = 16, kernel_size=(5, 3, 3), activation='relu')(conv1)
	conv3 = Conv3D(filters = 32, kernel_size=(3, 3, 3), activation='relu')(conv2)
	flatten_layer = Flatten()(conv3)
	dense1 = Dense(units = 256, activation = 'relu')(flatten_layer)
	dense1 = Dropout(0.4)(dense1)
	dense2 = Dense(units = 128, activation = 'relu')(dense1)
	dense2 = Dropout(0.4)(dense2)
	output_layer = Dense(units = numLabels, activation = 'softmax')(dense2)

	model= Model(inputs = input_layer, outputs = output_layer)
	model.summary()
	config = model.to_json()
	print(type(config))
	print(config)
	#quit()

######################################################################
######################################################################


	# Compile the model
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	#csv_logger = CSVLogger('log1.csv', separator=",", append=False) TODO logger not found 
	# Train the model
	model.fit(train_data, train_labels, epochs=101, batch_size=128, validation_data=(test_data, test_labels))#, callbacks=[csv_logger])

	# Evaluate the model
	test_loss, test_accuracy = model.evaluate(test_data, test_labels)
	print('Test Loss:', test_loss)
	print('Test Accuracy:', test_accuracy)

classify(data, test_data)
