import numpy as np
import csv

classesPath = "../data/datasets/classCSVs/"
projectionPath = "../data/channelProjection.npy"
outputDir = "../data/datasets/sequences"
file = "MM_RLH_T1.csv"
#csv_file = classesPath + file
output = []

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
'''
def processChunk(chunk):
    # print(str(len(chunk)/80))
    sequence_array = []
    # len(chunk)/80 is 9.0125 <- 721/80 but 720/80 = 9 (0.2 seconds at beginning is baseline anyway)
    chunk.pop(0)
    while chunk:
        print(f"size of chunk array {len(chunk)}")
        frame_array = []
        if len(chunk) == 80:
            while chunk:
                replace_indices_with_values(chunk.pop(), array)
            sequence_array.append(frame_array)
            break
        for index, x in enumerate(chunk):
            frame_array.append(replace_indices_with_values(chunk.pop(), array))
            if len(frame_array) == 80:
                sequence_array.append(frame_array)
                nicePrintNPFloat(frame_array[0])
                #print(f"length of frame_array: {len(frame_array)}")
                #print(f"length of sequence_array: {len(sequence_array)}")
                break
        #if len(frame_array) != 80:
            #print(f"while chunk exiting with frame_array at a length of {len(frame_array)}")
    #print(len(sequence_array))
    output.extend(sequence_array)
   
    #print(len(output))
def processChunk(chunk):
    sequence_array = []
    chunk.pop(0)
    while chunk:
        frame_array = []
        if len(chunk) == 80:
            while chunk:
                replace_indices_with_values(chunk.pop(), array)
            sequence_array.append(frame_array)
            break
        index = 0  # Initialize index outside the loop
        while index < len(chunk):
            frame_array.append(replace_indices_with_values(chunk.pop(), array))
            index += 1  # Increment the index manually
            if len(frame_array) == 80:
                sequence_array.append(frame_array)
                break
    output.extend(sequence_array)
'''
def processChunk(chunk):
    sequence_array = []
    chunk.pop(0)
    while chunk:
        frame_array = []
        if len(chunk) < 80:
            break
        for _ in range(80):
            frame_array.append(replace_indices_with_values(chunk.pop(0), array))
        sequence_array.append(frame_array)
    output.extend(sequence_array)


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
count = 0
def chunkEach(csv_file)
	csv_file = classesPath + csv_file
	with open(csv_file, 'r') as file:
	    reader = csv.DictReader(file)
	    rows = [row for row in reader]

	    # Group rows based on specified keys while preserving order
	    key_columns = ['subject', 'epoch', 'run']
	    row_groups = group_rows_preserve_order(rows, key_columns)

	    # Process each chunk
	    for chunk in row_groups:
		count = count + 1
		processChunk(chunk)

print(str(count))

for x in output:
	print(len(x))

def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            csv_files.append(filename)
    return csv_files

# Example usage
csv_files_list = get_csv_files(classesPath)

print(csv_files_list)
outputNp = np.array(output)

np.save('my_array.npy', outputNp)
