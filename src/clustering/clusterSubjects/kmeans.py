import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


path = "../../../data/datasets/processed7/sequences/MM_RLH_T1.npy"
pathannotations = "../../../data/datasets/processed7/sequences/MM_RLH_T1_annotations.csv"

data = np.load(path)

data = np.reshape(data, (2353, 1200, 17, 17))
print(data.shape)

data_min = np.min(data, axis=(1,2,3), keepdims=True)  # get min value for each group
data_max = np.max(data, axis=(1,2,3), keepdims=True)  # get max value for each group

# to avoid division by zero, add a small constant to the denominator
epsilon = np.finfo(float).eps
video_data = (data - data_min) / (data_max - data_min + epsilon)

print(f"shape video_data {video_data.shape}")

def cluster(n_clusters, num_iter):
    # Reshape the video data to 2D vectors
    n_videos, n_frames, width, height = video_data.shape
    video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

    # Create a KMeans instance
    kmeans = KMeans(n_clusters=n_clusters, max_iter=num_iter, n_init=10, random_state=42)

    # Train the KMeans
    kmeans.fit(video_data_2d)

    # Get the clustered labels for each video
    cluster_labels = kmeans.labels_

    print("#######################################################################################")
    # Calculate the Davies-Bouldin Index for each cluster
    dbi = davies_bouldin_score(video_data_2d, cluster_labels)
    print(f"Davies-Bouldin Index for {n_clusters} clusters: {dbi}")
    counts = np.bincount(cluster_labels.astype(int))
    print(counts)
    print("#######################################################################################")


for i in range(2, 20, 4):
    cluster(i, 500)
