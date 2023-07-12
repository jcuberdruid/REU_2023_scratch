#{'chunkIndex': '1', 'index': '0', 'subject': '3', 'run': '3', 'epoch': '3'}
import numpy as np
import csv

pathannotations = "../../../data/datasets/processed/sequences/MM_RLH_T2_annotation.csv"
pathNPY = "../../../data/datasets/processed/sequences/MM_RLH_T2.npy"

data = np.load(pathNPY)
#reshaped_array = data.reshape((33615, 80, 17, 17))

#np.save(pathNPY, reshaped_array)

print(data.shape)

# Select the first 17 elements of the first axis
selected_data = data[:17]

for index, x in enumerate(selected_data):
    for index2, y in enumerate(selected_data):
        if np.array_equal(x, y):
            print(f"subepoch {index2} equals subepoch {index}")

# Load annotation
annotations = []
with open(pathannotations, "r") as file:
    reader = csv.DictReader(file)
    rows_per_chunk = {}
    for row in reader:
        annotations.append(row)
        chunk_index = row['chunkIndex']
        if chunk_index in rows_per_chunk:
            rows_per_chunk[chunk_index] += 1
        else:
            rows_per_chunk[chunk_index] = 1

print(len(annotations))
quit()
# Check number of rows per chunk
for chunk_index, row_count in rows_per_chunk.items():
    if row_count != 15:
        print(f"Chunk {chunk_index} has {row_count} rows.")
    else:   
        print("ok")
