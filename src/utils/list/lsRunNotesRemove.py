import os
import glob
import json
import math
import numpy as np
import sys
import shutil

if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = None

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

def delete_files_with_run_note(files, run_note):
    for file in files:
        with open(file) as f:
            data = json.load(f)
            if 'run_note' in data and data['run_note'] == run_note:
                os.remove(file)
                print(f"File {file} removed.")

perfPath = '../../../logs/tf_perf_logs/'

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

if target is not None:
    delete_files_with_run_note(output_logs_array, target)
    quit()

print("#################################")
print("# Runs above 50")
print("#################################")

accuracy_data_list = []

for x in split:
    if len(x) > 2:
        accuracy_list = [y['accuracy'] for y in x]
        avg_accuracy = sum(accuracy_list) / len(accuracy_list)
        run_note = str(x[0]['run_note'])
        
        accuracy_data = {
            'len': len(x),
            'accuracy': avg_accuracy,
            'run_note': run_note
        }

        accuracy_data_list.append(accuracy_data)

sorted_accuracy_data = sorted(accuracy_data_list, key=lambda k: k['accuracy'])

for item in sorted_accuracy_data:
    print(f"{item['len']}, accuracy: {item['accuracy']}", item['run_note'].rjust(15))

