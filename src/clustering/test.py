import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

path = "../../data/datasets/processed4/sequences/MM_RLH_T1.npy"
pathannotations = "../../data/datasets/processed4/sequences/MM_RLH_T1_annotations.csv"

npyLoad = np.load(path)
print(npyLoad.shape)

video_data = npyLoad
print(f"shape video_data {video_data.shape}")


# Reshape the video data to 2D vectors and perform clustering
n_videos, n_frames, width, height = video_data.shape
video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

# Create a KMeans instance
kmeans = KMeans(n_clusters=2, random_state=0)

# Train the KMeans
kmeans.fit(video_data_2d)

# Get the clustered labels for each video
cluster_labels = kmeans.labels_

# Perform PCA to reduce the dimensions to 2D for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(video_data_2d)

# Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-Means Clustering Visualization with PCA')
plt.show()

