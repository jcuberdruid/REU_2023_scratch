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

accuracy = None
loss = None

# generalization: specific output classes
# output expirment data

csv_label_1 = "../../data/datasets/sequences/MI_RLH_T1_annotation.csv"
npy_label_1 = "../../data/datasets/sequences/MI_RLH_T1.npy"

csv_label_2 = "../../data/datasets/sequences/MI_RLH_T2_annotation.csv"
npy_label_2 = "../../data/datasets/sequences/MI_RLH_T2.npy"

training_files = [npy_label_1, npy_label_2]

csv_label_1_testing = "../../data/datasets/sequences/MI_RLH_T1_annotation.csv"
npy_label_1_testing = "../../data/datasets/sequences/MI_RLH_T1.npy"

csv_label_2_testing = "../../data/datasets/sequences/MI_RLH_T2_annotation.csv"
npy_label_2_testing = "../../data/datasets/sequences/MI_RLH_T2.npy"

testing_files = [npy_label_1_testing, npy_label_2_testing]

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


subjects = [x for x in range(1, 110) if x not in [88, 92, 100, 104]]

random.shuffle(subjects)

testingSubjects = []
testingSubjects.append(subjects.pop(0))

# testingSubjects = [subjects[0]]

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

test_data_1 = create_data(
    csv_label_1_testing, testingSubjects, npy_label_1_testing)
test_data_2 = create_data(
    csv_label_2_testing, testingSubjects, npy_label_2_testing)


#shuffle data into 90 and 10 percent (ish)
test_data_1_perc = np.concatenate((test_data_1[:0], test_data_1[len(test_data_1)-16:]))
test_data_1 = np.concatenate((test_data_1[:len(test_data_1)-16], test_data_1[len(test_data_1):]))

test_data_2_perc = np.concatenate((test_data_2[:0], test_data_2[len(test_data_2)-16:]))
test_data_2 = np.concatenate((test_data_2[:len(test_data_2)-16], test_data_2[len(test_data_2):]))

#concat the 10 percent of the subject into the testing data 
data_1_including_sub = np.concatenate((data_1, test_data_1_perc), axis=0)
data_2_including_sub = np.concatenate((data_2, test_data_2_perc), axis=0)


