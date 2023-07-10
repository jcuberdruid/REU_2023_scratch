import csv
import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#constants 
targetSubject = 8
csv_label_1 = "../../data/datasets/sequences/MI_RLH_T1_annotation.csv"
npy_label_1 = "../../data/datasets/sequences/MI_RLH_T1.npy"

epochs_annotation_array = []

with open(csv_label_1, 'r') as file:
    reader = csv.DictReader(file)
    current_chunk = 1
    this_epoch_array = []
    for index, row in enumerate(reader):
        if current_chunk != int(row['chunkIndex']):
            epochs_annotation_array.append(this_epoch_array)
            this_epoch_array = []  # Create a new empty list for the next chunk
            current_chunk = int(row['chunkIndex'])

        this_epoch_array.append(row)

if this_epoch_array:
    epochs_annotation_array.append(this_epoch_array)


#remove target subject from annotation list 
for index, x in enumerate(epochs_annotation_array):
    if (int(x[0]['subject']) == targetSubject):
        print(index)
        print(len(epochs_annotation_array.pop(index)))

#create epoch indices list  
epochs_indices = []
for index, x in enumerate(epochs_annotation_array):
    this_epoch_indices = []
    for y in x:
        this_epoch_indices.append(int(y['index']))
    epochs_indices.append(this_epoch_indices)

npyLoad = np.load(npy_label_1)
def load_epoch(indices):
    npySubSet = []
    max_value = max(indices)
    min_value = min(indices)
    for x in indices:
        npySubSet.append(npyLoad[x])
    #turn (9, 80, 17, 17) into (720, 17, 17)
    stackEpoch = np.stack(npySubSet, axis=0)
    concatEpoch = np.reshape(stackEpoch, (720, 17, 17))
    return concatEpoch

epochs = []
for x in epochs_indices:
    epochs.append(load_epoch(x))

print(len(epochs))
epochsStacked = np.stack(epochs, axis=0)
print(epochsStacked.shape)

##############################################
##############################################

video_data = epochsStacked

# Reshape the video data to 2D vectors
n_videos, n_frames, width, height = video_data.shape
video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

# Create a SOM instance
som = MiniSom(x=1, y=2, input_len=n_frames * width * height, sigma=0.3, learning_rate=0.5)

# Initialize the SOM
som.random_weights_init(video_data_2d)

# Train the SOM
som.train_random(video_data_2d, num_iteration=1000)

# Get the clustered labels for each video
cluster_labels = np.zeros(n_videos)
for i in range(n_videos):
    video_vector = video_data_2d[i]
    cluster_labels[i] = som.winner(video_vector)[-1]

print(cluster_labels)

counts = np.bincount((cluster_labels.astype(int)))
for value, count in enumerate(counts):
    print(f"Count of {value}: {count}")

print(f"the length of the cluster labels is {len(cluster_labels)} and the shape of the epochs is {epochsStacked.shape}")

clusteredEpochs = []
for index, x in enumerate(epochs_annotation_array):
    thisDict = {'subject':int(x[0]['subject']), 'epoch':int(x[0]['epoch']), 'cluster':int(cluster_labels[index])}
    clusteredEpochs.append(thisDict)



