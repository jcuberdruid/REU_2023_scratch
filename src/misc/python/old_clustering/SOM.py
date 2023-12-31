import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "../../data/datasets/hilowonly/sequences/MI_RLH_T1.npy"
epochs_annotation_array = []

npyLoad = np.load(path)
print(npyLoad.shape)

video_data = npyLoad
print(f"shape video_data {video_data.shape}")
# Reshape the video data to 2D vectors

n_videos, n_frames, width, height = video_data.shape
video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

# Create a SOM instance
som = MiniSom(x=1, y=2, input_len=n_frames * width * height, sigma=0.5, learning_rate=0.5)

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

for x in range(0, len(cluster_labels)):
    print(cluster_labels[x])

counts = np.bincount((cluster_labels.astype(int)))

for value, count in enumerate(counts):
    print(f"Count of {value}: {count}")

clusteredEpochs = []
for index, x in enumerate(epochs_annotation_array):
    thisDict = {'subject':int(x[0]['subject']), 'epoch':int(x[0]['epoch']), 'cluster':int(cluster_labels[index])}
    clusteredEpochs.append(thisDict)

for x in clusteredEpochs:
    print(x)
