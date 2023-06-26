import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

path = "../../data/datasets/sequences/MI_RLH_T1.npy"

npyLoad = np.load(path)
video_data = npyLoad[:250]

n_videos, n_frames, width, height = video_data.shape
video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

som = MiniSom(x=10, y=10, input_len=n_frames * width * height, sigma=1.0, learning_rate=0.5)

som.random_weights_init(video_data_2d)
som.train_random(video_data_2d, num_iteration=100)

cluster_labels = np.zeros(n_videos)
for i in range(n_videos):
    video_vector = video_data_2d[i]
    cluster_labels[i] = som.winner(video_vector)[-1]

unique_clusters = np.unique(cluster_labels)

color_map = plt.cm.get_cmap('tab20', len(unique_clusters))

plt.figure(figsize=(10, 10))
for i, video_vector in enumerate(video_data_2d):
    winner = som.winner(video_vector)
    plt.plot(winner[0] + 0.

