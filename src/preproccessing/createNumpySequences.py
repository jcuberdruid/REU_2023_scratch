import numpy as np
import os
import csv
from dataclasses import dataclass
import paths

classesPath = paths.dirBase+"classes/"
projectionPath = "../../data/channelProjection.npy"
outputDir = paths.dirBase+"sequences/"
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


def replace_indices_with_values(dictionary, array):
    float_array = np.zeros(array.shape, dtype=np.float64)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            index = array[i, j]
            if index in dictionary:
                float_array[i, j] = float(dictionary[index])
    return float_array


def processChunk(chunk):
    global chunkCount
    global count
    countNumberLoops = 0
    print(f"chunkCount {chunkCount}, should get to higher than 18k")
    print(f"length of chunk {len(chunk)}")
    #chunk.pop(0)
    chunkCount = chunkCount + 1
    sequence_array = []
    while chunk:
        chunkBuffer = chunk.copy()  # holds buffer 
        #print(f"length of chunk: {len(chunk)}")
        countNumberLoops = countNumberLoops + 1
        #print(countNumberLoops)
        frame_array = []
        if len(chunk) < 80:
            print("breaking")
            break
        for i, _ in enumerate(range(80)):
            chunkPopped = chunkBuffer.pop(0)
            frame_array.append(replace_indices_with_values(chunkPopped, array))
        annotation = AnnotationStruct(chunkIndex=chunkCount, index=count, subject=int(chunkBuffer[0]['subject']), run=int(chunkBuffer[0]['run']), epoch=int(chunkBuffer[0]['epoch']))
        count = count + 1
        chunkAnnotations.append(annotation)
        sequence_array.append(frame_array)
        for _ in range(40):
            chunk.pop(0)

    output.extend(sequence_array)


def write_annotation_csv(data: list, csv_name: str):
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
        #print(len(row_groups))
        #print(len(row_groups[0])) # 4 seconds
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
    print(f"proccessing file {x}")
    chunkEach(x)
    count = 0
    outputNp = np.array(output)
    x = x.split(".")[0]
    npy_path = outputDir+x+".npy"
    np.save(npy_path, outputNp)
    output.clear()
    write_annotation_csv(chunkAnnotations, os.path.splitext(npy_path)[0] + "_annotation.csv")
    chunkAnnotations.clear()
    chunkCount = 0
