import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "../../data/datasets/processed7/sequences/MM_RLH_T1.npy"
epochs_annotation_array = []

npyLoad = np.load(path)
#npyLoad = np.reshape(npyLoad, (2353, 1200, 17, 17))

print(npyLoad.shape)

video_data = npyLoad
print(f"shape video_data {video_data.shape}")

# Reshape the video data to 2D vectors
n_videos, n_frames, width, height = video_data.shape
video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

# Standardize the data
scaler = StandardScaler()
video_data_2d = scaler.fit_transform(video_data_2d)

# Assume that the first 300 videos are the target videos

print(f"length of all videos {len(video_data)}")
target_videos = video_data_2d[:100]
print(f"length of :300 {len(target_videos)}")
other_videos = video_data_2d[100:30000]
print(f"length of 3000: {len(other_videos)}")

# Create a NearestNeighbors instance
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine')

# Fit the NearestNeighbors instance on the target videos
nbrs.fit(target_videos)

# Get the distances and indices of the nearest target video for each other video
distances, indices = nbrs.kneighbors(other_videos)

# Sort the other videos by their distances to the nearest target video
sorted_indices = np.argsort(distances, axis=0)
sorted_other_videos = other_videos[sorted_indices]

for x in sorted_indices:
    print(x)

print("####")
print(len(sorted_indices))
