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
som = MiniSom(x=10, y=10, input_len=n_frames * width * height, sigma=1.0, learning_rate=0.5)

# Initialize the SOM
som.random_weights_init(video_data_2d)

# Train the SOM
som.train_random(video_data_2d, num_iteration=100)

# Get the clustered labels for each video
cluster_labels = np.zeros(n_videos)
for i in range(n_videos):
    video_vector = video_data_2d[i]
    cluster_labels[i] = som.winner(video_vector)[-1]

# Create a scatter plot of the 2D video locations
plt.figure(figsize=(8, 6))
for label in np.unique(cluster_labels):
    videos_in_cluster = video_data_2d[cluster_labels == label]
    x = videos_in_cluster[:, 0]  # Extract X coordinates
    y = videos_in_cluster[:, 1]  # Extract Y coordinates
    plt.scatter(x, y, label=f'Cluster {label}')

plt.title('Video Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('test.png')
plt.show()


