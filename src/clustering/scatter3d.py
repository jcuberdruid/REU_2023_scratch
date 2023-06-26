import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "../../data/datasets/sequences/MI_RLH_T1.npy"

npyLoad = np.load(path)
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

# Apply PCA for dimensionality reduction to 3D
pca = PCA(n_components=3)
video_data_3d_pca = pca.fit_transform(video_data_2d)

# Create a 3D scatter plot of the video locations
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(cluster_labels):
    videos_in_cluster = video_data_3d_pca[cluster_labels == label]
    x = videos_in_cluster[:, 0]  # Extract X coordinates
    y = videos_in_cluster[:, 1]  # Extract Y coordinates
    z = videos_in_cluster[:, 2]  # Extract Z coordinates
    ax.scatter(x, y, z, label=f'Cluster {label}')

ax.set_title('Video Clustering')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.show()

