import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

path = "../../data/datasets/processed4/sequences/MM_RLH_T1.npy"
pathannotations = "../../data/datasets/processed4/sequences/MM_RLH_T1_annotations.csv"


annotations = []
with open(pathannotations, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            annotations_arr.append(row)
    annotations.append(annotations_arr)

print(annotations[0])
quit()
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

    return cluster_labels, video_data_2d

print("#######################################################################################")
cluster_labels, video_data_2d = kmeans_clustering(video_data, 10)
#remove smaller bin
counts = np.bincount(cluster_labels.astype(int))
smaller_cluster_label = np.argmin(counts)
mask = cluster_labels != smaller_cluster_label
video_data_2d = video_data_2d[mask]
cluster_labels = cluster_labels[mask]
print(counts)

# Apply PCA first for dimension reduction
pca = PCA(n_components=50)
video_data_pca = pca.fit_transform(video_data_2d)

# Then apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
video_data_tsne = tsne.fit_transform(video_data_pca)

# Plot the t-SNE output
plt.figure(figsize=(10, 8))
scatter = plt.scatter(video_data_tsne[:, 0], video_data_tsne[:, 1], c=cluster_labels, s=50, cmap='viridis')

# Create a legend for the clusters
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

plt.show()

