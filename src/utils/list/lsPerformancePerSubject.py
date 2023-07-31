# This script will list the different "runNotes" in the output.json files, and how many files there are with each note
import os
import glob
import json
import math
import numpy as np
import sys


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
print("# Runs above 50")
print("#################################")

length = 105
highest_accuracy = [0] * length
run_notes = ["0"] * length

for x in split:
    if len(x) != 103:
        accuracy_list = [y['accuracy'] for y in x]
        avg_accuracy = sum(accuracy_list) / len(accuracy_list)
        for index, y in enumerate(x):
            if highest_accuracy[index] < float(y['accuracy']):
                highest_accuracy[index] = float(y['accuracy'])
                if 'run_note' in y:
                    run_notes[index] = (str(y['run_note']))
                else: 
                    print("unknown")
                    run_notes.append("Uknown")
        



for index, x in enumerate(highest_accuracy):
    print(f"{index} : {x} : {run_notes[index]}")

#highest_accuracy.pop(len(highest_accuracy)-1)

print(sum(highest_accuracy)/len(highest_accuracy))


target_run_note = sorted(highest_accuracy)  # Sort the data in ascending order

print(target_run_note)

middle_80_percent = target_run_note[int(len(target_run_note) * 0.1):int(len(target_run_note) * 0.9)]

middle_80_avg = sum(middle_80_percent) / len(middle_80_percent)
print("Average (Middle 80%):", middle_80_avg)

# Calculate average for upper 10%
upper_10_percent = target_run_note[int(len(target_run_note) * 0.9):]
upper_10_avg = sum(upper_10_percent) / len(upper_10_percent)
print("Average (Upper 10%):", upper_10_avg)

avg = sum(target_run_note) / len(target_run_note)
print("Average (Overall):", avg)
