import numpy as np
import csv

csv_label_1 = "../data/datasets/sequences/MI_RLH_T1_annotation.csv"
npy_label_1 = "../data/datasets/sequences/MI_RLH_T1.npy"

csv_label_2 = "../data/datasets/sequences/MI_RLH_T2_annotation.csv"
npy_label_2 = "../data/datasets/sequences/MI_RLH_T2.npy"

thisSubject = 49

def get_indices_for_subject(csv_file, subject):
    indices = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['subject']) == subject:  # Convert subject to integer
                indices.append(int(row['index']))
    return indices

def data_for_subject(npy_file, indices):
    npySubSet = []
    npyLoad = np.load(npy_file)
    for x in indices:
        npySubSet.append(npyLoad[x])
    return npySubSet


indices_label_1 = get_indices_for_subject(csv_label_1, thisSubject)
indices_label_2 = get_indices_for_subject(csv_label_2, thisSubject)

npyData_label_1 = data_for_subject(npy_label_1, indices_label_1)
npyData_label_2 = data_for_subject(npy_label_2, indices_label_2)

print(indices_label_1)
print(len(npyData_label_1))
print(indices_label_2)
print(len(npyData_label_2))

