import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

path = "../../data/datasets/sequences/MI_RLH_T1.npy"

npyLoad = np.load(path)
print(npyLoad[:5])
print(npyLoad.shape)

video_data = npyLoad[:250]

# Reshape the video data to 2D vectors
n_videos, n_frames, width, height = video_data.shape
video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

# Create a SOM instance
som = MiniSom(x=100, y=100, input_len=n_frames * width * height, sigma=1.0, learning_rate=0.5)

# Initialize the SOM
som.random_weights_init(video_data_2d)

# Train the SOM
som.train_random(video_data_2d, num_iteration=100)

# Get the clustered labels for each video
cluster_labels = np.zeros(n_videos)
for i in range(n_videos):
    video_vector = video_data_2d[i]
    cluster_labels[i] = som.winner(video_vector)[-1]

# Now you have the cluster labels for each video
print(cluster_labels)

unique_clusters = np.unique(cluster_labels)

print(unique_clusters)

# Generate a grid of colors based on the cluster labels
color_map = plt.cm.get_cmap('tab10', len(unique_clusters))

# Plot the SOM grid and color the neurons based on cluster labels
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Plot the distance map as background
for i, video_vector in enumerate(video_data_2d):
    winner = som.winner(video_vector)
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, 'o', markerfacecolor=color_map(cluster_labels[i] / len(unique_clusters)), markersize=2, markeredgecolor='k')
plt.xticks(np.arange(0, 100, 10))
plt.yticks(np.arange(0, 100, 10))
plt.grid(True)
plt.savefig("test.png")

