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
print("# Runs above 10")
print("#################################")

target_run_note = [] 
for x in split:
    if len(x) > 10:
        s = str(x[0]['run_note'])
        if target != None:
            if s == target:
                for y in x:
                    target_run_note.append(float(y['accuracy']))

        print(f"{len(x)}", s.rjust(15)) 

if target == None:
    quit()

print("#################################")
print(f"# {target} ")
print("#################################")


target_run_note.sort()  # Sort the data in ascending order

# Calculate average for lower 10%
lower_10_percent = target_run_note[:int(len(target_run_note) * 0.1)]
lower_10_avg = sum(lower_10_percent) / len(lower_10_percent)
print("Average (Lower 10%):", lower_10_avg)

# Calculate average for middle 80%
middle_80_percent = target_run_note[int(len(target_run_note) * 0.1):int(len(target_run_note) * 0.9)]
middle_80_avg = sum(middle_80_percent) / len(middle_80_percent)
print("Average (Middle 80%):", middle_80_avg)

# Calculate average for upper 10%
upper_10_percent = target_run_note[int(len(target_run_note) * 0.9):]
upper_10_avg = sum(upper_10_percent) / len(upper_10_percent)
print("Average (Upper 10%):", upper_10_avg)

avg = sum(target_run_note) / len(target_run_note)
print("Average (Overall):", avg)

variance = sum((x - avg) ** 2 for x in target_run_note) / len(target_run_note)
print("Variance:", variance)

data = np.array(target_run_note)
# Calculate quartiles
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
# Calculate interquartile range
iqr = q3 - q1
print("Interquartile range:", iqr)
