import pandas as pd
import re
import os

directory_path = "../../data/datasets/unproccessed/tasks/"
output_path = "../../data/datasets/unproccessed/classes" 

def get_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def merge_csvs(files, file_name):
    df = pd.DataFrame()
    for file in files:
        data = pd.read_csv(file)
        df = pd.concat([df, data], axis=0)
    df.to_csv(file_name, index=False)

paths_array = get_file_paths(directory_path)
print(len(paths_array))

patterns = [
    "MI_RLH_T1",
    "MI_RLH_T2",
    "MM_RLH_T1",
    "MM_RLH_T2",
    "MI_FF_T1",
    "MI_FF_T2",
    "MM_FF_T1",
    "MM_FF_T2"
]

sorted_arrays = [[] for _ in range(len(patterns))]

for file_path in paths_array:
    file_name = os.path.basename(file_path)
    for i, pattern in enumerate(patterns):
        if pattern in file_name:
            sorted_arrays[i].append(file_path)
            break

# Print the sorted arrays
for i, pattern in enumerate(patterns):
    print(f"Files matching pattern '{pattern}':")
    print(len(sorted_arrays[i]))
    file_name = output_path + "/" + pattern + ".csv"
    merge_csvs(sorted_arrays[i], file_name)
    print()

