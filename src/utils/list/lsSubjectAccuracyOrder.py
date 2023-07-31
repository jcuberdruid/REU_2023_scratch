# This script will list the different "runNotes" in the output.json files, and how many files there are with each note
import os
import glob
import json
import math
import numpy as np
import sys
from scipy.stats import spearmanr

def calculate_order_similarity(arr):
    ranked_lists = []
    for sublist in arr:
        temp_dict = {value : rank for rank, value in enumerate(sorted(set(sublist)), 1)}
        ranked_list = [temp_dict[i] for i in sublist]
        ranked_lists.append(ranked_list)
        
    # Calculate similarity between all pairs of arrays
    similarity_matrix = []
    for i in range(len(ranked_lists)):
        row = []
        for j in range(len(ranked_lists)):
            coefficient, _ = spearmanr(ranked_lists[i], ranked_lists[j])
            row.append(coefficient)
        similarity_matrix.append(row)
    
    return similarity_matrix

if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = None


# gets all the directories in a given path and returns them as an array  
def list_directories(path):
    directories = []
    for entry in os.listdir(path):
        if os.path.isdir(os.path.join(path, entry)):
            directories.append(entry)
    return directories

def split_dicts_by_key(arr, key):
    result = {}
    for item in arr:
        if key in item:
            value = item[key]
            if value not in result:
                result[value] = []
            result[value].append(item)

    arrays = list(result.values())
    return arrays

# paths
perfPath = '../../../logs/tf_perf_logs/'

# output_logs_array
output_logs_array = []

directory_list = list_directories(perfPath)
for directory in directory_list:
    pattern = perfPath+directory+'/output_log*'
    output_logs_array.extend(glob.glob(pattern))


output_logs_json = []
for x in output_logs_array:
    with open(x) as file:
        output_logs_json.append(json.load(file))

split = split_dicts_by_key(output_logs_json, "run_note")


print("#################################")
print("# Runs above 10")
print("#################################")

accuracy_data_list = []
accuracy_run_note = []
for x in split:
    if len(x) == 103:
        accuracy_list = [y['accuracy'] for y in x]
        avg_accuracy = sum(accuracy_list) / len(accuracy_list)
        run_note = str(x[0]['run_note'])

        x = sorted(x, key=lambda k: k['accuracy'])
        subjectsByAccuracy = [int(y['testing_subjects'][0]) for y in x]
        accuracy_data_list.append(subjectsByAccuracy)

        accuracy_data = {
            'len': len(x),
            'accuracy': avg_accuracy,
            'run_note': run_note
        }

        accuracy_run_note.append(accuracy_data)

sorted_accuracy_data = sorted(accuracy_run_note, key=lambda k: k['accuracy'])

for item in sorted_accuracy_data:
    print(f"{item['len']}, accuracy: {item['accuracy']}", item['run_note'].rjust(15))


def print_matrix(matrix):
    for row in matrix:
        print(" ".join("{:0.2f}".format(cell) for cell in row))

print_matrix(calculate_order_similarity(accuracy_data_list))
