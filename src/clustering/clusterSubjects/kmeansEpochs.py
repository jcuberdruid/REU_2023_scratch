import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


path1 = "../../../data/datasets/processed7/sequences/MM_RLH_T1.npy"
path2 = "../../../data/datasets/processed7/sequences/MM_RLH_T2.npy"
pathannotations = "../../../data/datasets/processed7/sequences/MM_RLH_T1_annotations.csv"

# ...
npyLoad1 = np.load(path1)
npyLoad1_len = len(npyLoad1) # keep track of the number of videos in file1
print(npyLoad1.shape)

npyLoad2 = np.load(path2)
npyLoad2_len = len(npyLoad2) # keep track of the number of videos in file2
print(npyLoad2.shape)

combined_arrays = np.vstack((npyLoad1, npyLoad2))

# Create masks for videos from each file
mask1 = np.array([True]*npyLoad1_len + [False]*npyLoad2_len)
mask2 = ~mask1

video_data = combined_arrays
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
    
    # Count cluster instances from each file
    counts_file1 = np.bincount(cluster_labels[mask1].astype(int))
    counts_file2 = np.bincount(cluster_labels[mask2].astype(int))

    # Calculate and print percentages
    percentages_file1 = (counts_file1 / npyLoad1_len) * 100
    percentages_file2 = (counts_file2 / npyLoad2_len) * 100

    print(f"Percentages of each cluster from file1: {percentages_file1}")
    print(f"Percentages of each cluster from file2: {percentages_file2}")
    print("#######################################################################################")

cluster(2, 1000)

