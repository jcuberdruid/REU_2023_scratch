import numpy as np
import csv
from collections import defaultdict
from keras.models import Model
from keras.layers import Dense, Input
#from DEC import DEC # You might need to install this using pip install DEC-keras
#from DEC-keras.DEC import DEC # assuming DEC-keras is a sibling directory of your current working directory
import sys
sys.path.insert(0, '/Users/eos/programming/reu_2023/src/clustering/dec/DEC-keras')

from DEC import DEC

import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load annotations
pathannotations = "../../../data/datasets/processed4/sequences/MM_RLH_T2_annotation.csv"
annotations = []
with open(pathannotations, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Convert index values to integers
        row['index'] = int(row['index'])
        annotations.append(row)

# Load npyLoad array
path = "../../../data/datasets/processed4/sequences/MM_RLH_T2.npy"
npyLoad = np.load(path)

# Normalize npyLoad on a per-chunk basis
def prepareData(chunk):
    # Reshape the chunk to 2D
    chunk_2d = chunk.reshape((-1, chunk.shape[-1]))
    # Normalize the chunk
    scaler = StandardScaler()
    chunk_normalized = scaler.fit_transform(chunk_2d)
    # Reshape the chunk back to 3D
    chunk_normalized = chunk_normalized.reshape(chunk.shape)
    return chunk_normalized

for annotation in annotations:
    index = annotation['index']
    chunk = npyLoad[index]

    # Normalize the chunk
    normalized_chunk = prepareData(chunk)

    # Update the normalized chunk in npyLoad
    npyLoad[index] = normalized_chunk

print(npyLoad)


# Assuming video_data is your input matrix
n_videos, n_frames, width, height = npyLoad.shape
video_data_2d = npyLoad.reshape(n_videos, n_frames * width * height)

# Let's define an autoencoder architecture
input_data = Input(shape=(n_frames * width * height,))
encoded = Dense(128, activation='relu')(input_data)
encoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(n_frames * width * height, activation='sigmoid')(decoded)

autoencoder = Model(input_data, decoded)
#autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='kld')


# Let's train the autoencoder
autoencoder.fit(video_data_2d, video_data_2d, epochs=50)

# Let's take the encoder part
encoder = Model(input_data, encoded)

# Now let's use DEC for clustering
dec = DEC(dims=[n_frames * width * height, 128, 64], n_clusters=20)
dec.pretrain(x=video_data_2d, y=None, optimizer='adam', epochs=10, batch_size=256)
dec.compile(optimizer='adam', loss='kld')
y_pred = dec.fit(video_data_2d)

# Now we use the predicted cluster labels in the rest of your script
cluster_labels = y_pred

# Remove smaller bin
counts = np.bincount(cluster_labels.astype(int))
smaller_cluster_label = np.argmin(counts)

print(counts)

# Generate a dictionary to hold count of data samples per cluster per subject
cluster_subject_dict = defaultdict(lambda: defaultdict(int))
# Dictionary to count total samples for each subject
subject_totals = defaultdict(int)

# Iterate through annotations and update the dictionary
for i, ann in enumerate(annotations):
    cluster_subject_dict[cluster_labels[i]][ann['subject']] += 1
    subject_totals[ann['subject']] += 1

# Calculate and print the percentage of each subject's data in each cluster
for cluster, subjects in cluster_subject_dict.items():
    print(f"Cluster {cluster} (total subjects: {len(subjects)}):")

    # Calculate percentage for each subject and sort the subjects based on the percentage
    subject_percentages = {subject: (count / subject_totals[subject]) * 100 for subject, count in subjects.items()}
    sorted_subjects = sorted(subject_percentages.items(), key=lambda item: item[1], reverse=True)

    for subject, percentage in sorted_subjects:
        print(f"    Subject {subject}: {percentage:.2f}% of its data in this cluster")

