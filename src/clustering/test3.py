import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

path = "../../data/datasets/sequences/MI_RLH_T1.npy"

npyLoad = np.load(path)
print(npyLoad[:5])
print(npyLoad.shape)

video_data = npyLoad

# Reshape the video data to 2D vectors
n_videos, n_frames, width, height = video_data.shape
video_data_2d = video_data.reshape(n_videos, n_frames, width * height)

# Flatten the 2D vectors
video_data_flattened = video_data_2d.reshape(n_videos * n_frames, width * height)

# Create a SOM instance
som = MiniSom(x=10, y=10, input_len=width * height, sigma=1.0, learning_rate=0.5)

# Initialize the SOM
som.random_weights_init(video_data_flattened)

# Train the SOM
som.train_random(video_data_flattened, num_iteration=100)

# Get the clustered labels for each video
cluster_labels = np.zeros(n_videos)
for i in range(n_videos):
    video_vectors = video_data_2d[i]
    bmu_positions = np.array([som.winner(vector.flatten())[-1] for vector in video_vectors])
    cluster_labels[i] = np.bincount(bmu_positions).argmax()

# Now you have the cluster labels for each video
print(cluster_labels)

unique_clusters = list(set(cluster_labels))

print(unique_clusters)

# Generate a grid of colors based on the cluster labels
color_map = plt.cm.get_cmap('tab10')

# Plot the SOM grid and color the neurons based on cluster labels
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Plot the distance map as background
for i, video_vectors in enumerate(video_data_2d):
    winners = np.array([som.winner(vector.flatten()) for vector in video_vectors])
    plt.plot(winners[:, 0] + 0.5, winners[:, 1] + 0.5, 'o', markerfacecolor=color_map(cluster_labels[i] / len(np.unique(cluster_labels))), markersize=8, markeredgecolor='k')
plt.xticks(np.arange(0, 10, 1))
plt.yticks(np.arange(0, 10, 1))
plt.grid(True)
plt.savefig("test.png")

