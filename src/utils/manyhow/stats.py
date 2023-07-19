###############################################################
# E 1: DWT correlation
# Q: Is there a closer DWT between electrodes during IM than during Rest+IM
# Y: this should tell us(hopefully) that there is a correlation between data in the IM tasks
# Y2: additionally the electrodes with closter DWT during the IM tasks vs Rest+IM tasks might be the electrodes with the best data
###############################################################
# I/O
# I: csv of a class? (how do I get rest data) is the rest+IM valid as rest might have super high DWT closeness?
# O: dwt closeness per electrode pair? (form groups based on DWT closeness?)

import csv
import numpy as np
from ctypes import *

# load the shared library
dtw_lib = CDLL('./dtw.so')  # or dtw.dll on Windows
dtw_lib.dtw.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_double)]
dtw_lib.dtw.restype = c_double

# Imports

# Paths and globals
csv_file_path = '/home/jc/keras/data/datasets/processed7/classes/MI_RLH_T1.csv'
target_subject = 23

# functions


def dtw(s, t):  # takes in numpy arrays
    # convert numpy arrays to ctypes arrays
    s_ctypes = s.ctypes.data_as(POINTER(c_double))
    t_ctypes = t.ctypes.data_as(POINTER(c_double))

    # get the lengths of the arrays
    n = len(s)
    m = len(t)

    # call the C function
    result = dtw_lib.dtw(n, m, s_ctypes, t_ctypes)

    return result

#take in csv


ts_data = []
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if int(row['subject']) == target_subject:
            ts_data.append(row)

print(len(ts_data))

FC5 = []
FC3 = []
IZ = []

for x in ts_data:
    FC5.append(float(x['FC5']))
    FC3.append(float(x['FC3']))
    IZ.append(float(x['IZ']))

# Convert lists to numpy arrays
FC5_np = np.array(FC5)
FC3_np = np.array(FC3)
IZ_np = np.array(IZ)

print(f"FC5 x FC3 DTW: {dtw(FC5_np, FC3_np)}")
print(f"FC5 x IZ DTW: {dtw(FC5_np, IZ_np)}")

print("#######################")
# Define the keys
keys = ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
        'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ',
        'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7',
        'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7',
        'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ']

# Initialize a dictionary to hold the data for each key
data_dict = {key: [] for key in keys}

# Populate the dictionary with data from ts_data
for x in ts_data:
    for key in keys:
        data_dict[key].append(float(x[key]))

# Convert lists in the dictionary to numpy arrays
data_dict = {key: np.array(val) for key, val in data_dict.items()}

# Generate all unique pairs of keys
from itertools import combinations

key_pairs = list(combinations(keys, 2))

pairing_results = []
# Perform DTW on each unique pair of keys
for pair in key_pairs:
    key1, key2 = pair
    dtw_result = dtw(data_dict[key1], data_dict[key2])
    saveDict = {'key1':key1,'key2':key2,'result':dtw_result}
    pairing_results.append(saveDict)
    print(f"{key1} x {key2} DTW: {dtw_result}")

data = pairing_results.copy()

keys = data[0].keys()  # Assuming all dictionaries have the same keys

filename = "data.csv"

with open(filename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=keys)
    writer.writeheader()  # Write the header row with the keys
    writer.writerows(data)  # Write the data rows
