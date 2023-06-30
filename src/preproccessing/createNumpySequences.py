import numpy as np
import os
import csv
from dataclasses import dataclass

classesPath = "../../data/datasets/unproccessed/classes/"
projectionPath = "../../data/channelProjection.npy"
outputDir = "../../data/datasets/unproccessed/sequences"
#file = "MM_RLH_T1.csv"
# csv_file = classesPath + file
output = []
chunkAnnotations = []
chunkCount = 0
count = 0

@dataclass
class AnnotationStruct:
    chunkIndex: int
    index: int
    subject: int
    run: int
    epoch: int


def nicePrintNP(arr):
    padding = 2
    max_length = max(len(element) for row in arr for element in row)
    format_str = '{:<' + str(max_length + padding) + '}'
    for row in arr:
        for element in row:
            print(format_str.format(element), end='')
        print()


def nicePrintNPFloat(arr):
    padding = 2
    max_length = max(len(str(element)) for row in arr for element in row)
    format_str = '{:<' + str(max_length + padding) + '}'
    for row in arr:
        for element in row:
            print(format_str.format(element), end='')
        print()


def replace_indices_with_values(dictionary, array):
    float_array = np.zeros(array.shape, dtype=np.float64)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            index = array[i, j]
            if index in dictionary:
                float_array[i, j] = float(dictionary[index])
    return float_array


# XXX add overlap here: (buffer of 10 added back to head of chunk list for 90-10 overlap)
def processChunk(chunk):
    global chunkCount
    global count
    chunkCount=chunkCount+1
    sequence_array = []
    chunk0 = chunk.pop(0)
    while chunk:
        frame_array = []
        if len(chunk) < 80:
            break
        for _ in range(80):
            frame_array.append(
                replace_indices_with_values(chunk.pop(0), array))
        annotation = AnnotationStruct(chunkIndex=chunkCount, index=count, subject=int(chunk0['subject']), run=int(chunk0['run']), epoch=int(chunk0['epoch'])) 
        count = count + 1
        chunkAnnotations.append(annotation)
        sequence_array.append(frame_array)
    output.extend(sequence_array)

def write_annoation_csv(data: list, csv_name: str):
    fieldnames = data[0].__dict__.keys()

    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            writer.writerow(item.__dict__)

# Function to group rows based on specified keys while preserving order
def group_rows_preserve_order(rows, keys):
    groups = []
    current_group = []
    prev_key_values = None

    for row in rows:
        key_values = [row[key] for key in keys]

        if key_values != prev_key_values and current_group:
            groups.append(current_group)
            current_group = []

        current_group.append(row)
        prev_key_values = key_values

    if current_group:
        groups.append(current_group)

    return groups


array = np.load(projectionPath)


def chunkEach(csv_file):
    csv_file = classesPath + csv_file
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]

        # Group rows based on specified keys while preserving order
        key_columns = ['subject', 'epoch', 'run']
        row_groups = group_rows_preserve_order(rows, key_columns)

        # Process each chunk
        for chunk in row_groups:
            processChunk(chunk)


def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            csv_files.append(filename)
    return csv_files


csv_files_list = get_csv_files(classesPath)

for x in csv_files_list:
    chunkEach(x)
    count = 0
    outputNp = np.array(output)
    x = x.split(".")[0]
    npy_path = outputDir+x+".npy"
    np.save(npy_path, outputNp)
    output.clear()
    write_annoation_csv(chunkAnnotations, os.path.splitext(npy_path)[0] + "_annotation.csv")
    chunkAnnotations.clear()
    chunkCount = 0
