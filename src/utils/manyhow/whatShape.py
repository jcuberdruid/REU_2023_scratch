import numpy as np
import os

# List of file paths
pathNPY = "../../../data/datasets/processed7/sequences/"

def print_npy_info(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            file_path = os.path.join(directory, filename)
            try:
                array = np.load(file_path)
                print(f"File: {filename}")
                print(f"Shape: {array.shape}")
                print()
            except Exception as e:
                print(f"Error loading file {filename}: {str(e)}")
                print()

# Example usage:
print_npy_info(pathNPY)

