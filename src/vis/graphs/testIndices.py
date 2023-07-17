import json
import matplotlib.pyplot as plt
import os
import hashlib
import pandas as pd
import seaborn as sns
import numpy as np

# Define the directory where your JSON files are located
directory = '../../../clustering_logs/processed3'

def get_similar_indices(json_filename):
    with open(json_filename, 'r') as json_file:
        json_data = json.load(json_file)

    class_key = f"class_1"
    if class_key not in json_data:
        return 0  # Return an empty list if the class number is not found

    class_info = json_data[class_key]
    similar_indices = class_info.get("similarIndices", [])

    return len(similar_indices)


def get_array_signature_strings(arr):
    # Sort the array to ensure consistent order
    sorted_arr = sorted(arr)
    # Convert the sorted array to a string
    arr_str = ','.join(sorted_arr)
    # Compute the hash of the array string
    signature = hashlib.md5(arr_str.encode()).hexdigest()
    return signature

def get_array_signature(arr):
    # Sort the array to ensure consistent order
    sorted_arr = sorted(arr)

    # Convert the sorted array to a string
    arr_str = ','.join(map(str, sorted_arr))

    # Compute the hash of the array string
    signature = hashlib.md5(arr_str.encode()).hexdigest()

    return signature

# Initialize an empty DataFrame to store the data from JSON files
data = []

for filename in os.listdir(directory):
	file_path = os.path.join(directory, filename)
	tmp = get_similar_indices(file_path)	
	data.append(tmp)

print(len(data))

# Calculate the average of 'data'
average_data = np.mean(data)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Create a bar plot with the average of 'data'
ax.bar('Average Size of Winning Cluster', average_data)

total_sequences = 35625
ax.bar('Total Sequences per Class', total_sequences)

# Set labels
ax.set_xlabel('Group')
ax.set_ylabel('Average Value')

# Show the plot
plt.savefig("sizing", dpi=300)
plt.show()

