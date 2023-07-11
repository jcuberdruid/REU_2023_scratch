import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kmeansCython import kmeans_clustering

path = "../../data/datasets/processed4/sequences/MM_RLH_T1.npy"
pathannotations = "../../data/datasets/processed4/sequences/MM_RLH_T1_annotations.csv"

npyLoad = np.load(path)
print(npyLoad.shape)

video_data = npyLoad
print(f"shape video_data {video_data.shape}")

cluster_labels, video_data_2d = kmeans_clustering(video_data, 10)

print("#######################################################################################")
#remove smaller bin
counts = np.bincount(cluster_labels.astype(int))
smaller_cluster_label = np.argmin(counts)
mask = cluster_labels != smaller_cluster_label
video_data = video_data[mask]
print(counts)

