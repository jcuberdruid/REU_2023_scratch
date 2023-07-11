import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "../../data/datasets/processed4/sequences/MM_RLH_T1.npy"
pathannotations = "../../data/datasets/processed4/sequences/MM_RLH_T1_annotations.csv"

npyLoad = np.load(path)
print(npyLoad.shape)

video_data = npyLoad
print(f"shape video_data {video_data.shape}")

# Reshape the video data to 2D vectors
def kmeans_clustering(video_data, clusters):
    n_videos, n_frames, width, height = video_data.shape
    video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

    # Create a KMeans instance
    kmeans = KMeans(n_clusters=clusters, random_state=0)

    # Train the KMeans
    kmeans.fit(video_data_2d)

    # Get the clustered labels for each video
    cluster_labels = kmeans.labels_

    return cluster_labels

print("#######################################################################################")
cluster_labels = kmeans_clustering(video_data, 3)
#remove smaller bin
counts = np.bincount(cluster_labels.astype(int))
smaller_cluster_label = np.argmin(counts)
mask = cluster_labels != smaller_cluster_label
video_data = video_data[mask]
print(counts)

