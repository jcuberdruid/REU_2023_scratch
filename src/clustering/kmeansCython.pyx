import numpy as np
from sklearn.cluster import KMeans

def kmeans_clustering(video_data, clusters):
    n_videos, n_frames, width, height = video_data.shape
    video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

    # Create a KMeans instance
    kmeans = KMeans(n_clusters=clusters, random_state=0)

    # Train the KMeans
    kmeans.fit(video_data_2d)

    # Get the clustered labels for each video
    cluster_labels = kmeans.labels_

    return cluster_labels, video_data_2d

