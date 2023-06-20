import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers


csv_label_1 = "../data/datasets/sequences/MI_RLH_T1_annotation.csv"
npy_label_1 = "../data/datasets/sequences/MI_RLH_T1.npy"

csv_label_2 = "../data/datasets/sequences/MI_RLH_T2_annotation.csv"
npy_label_2 = "../data/datasets/sequences/MI_RLH_T2.npy"

thisSubject = 9

def get_indices_for_subject(csv_file, subject):
    indices = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['subject']) == subject:  # Convert subject to integer
                indices.append(int(row['index']))
    return indices

def data_for_subject(npy_file, indices):
    npySubSet = []
    npyLoad = np.load(npy_file)
    for x in indices:
        npySubSet.append(npyLoad[x])
    return npySubSet


indices_label_1 = get_indices_for_subject(csv_label_1, thisSubject)
indices_label_2 = get_indices_for_subject(csv_label_2, thisSubject)

npyData_label_1 = np.array(data_for_subject(npy_label_1, indices_label_1))
npyData_label_2 = np.array(data_for_subject(npy_label_2, indices_label_2))

np.save('label_1.npy', npyData_label_1)
np.save('label_2.npy', npyData_label_2)

###############################################################
###############################################################


# Load label_1 and label_2 data from .npy files
data_label_1 = np.load('label_1.npy')
data_label_2 = np.load('label_2.npy')

# Concatenate data and labels
data = np.concatenate([data_label_1, data_label_2], axis=0)
labels = np.concatenate([np.zeros(data_label_1.shape[0]), np.ones(data_label_2.shape[0])], axis=0)

# Reshape the data to 2D
data_2d = data.reshape((-1, data.shape[-1]))

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_2d)

# Reshape the data back to 4D
data_normalized = data_normalized.reshape(data.shape)

# Shuffle the order of frames within each sample
np.random.shuffle(data_normalized)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data_normalized, labels, test_size=0.1, random_state=42)

# Reshape the data to match the input shape for the 3D-CNN
train_data = train_data.reshape((-1, 80, 17, 17, 1))
test_data = test_data.reshape((-1, 80, 17, 17, 1))

###############################################################
###############################################################

# Define the model
model = tf.keras.Sequential([
    layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(80, 17, 17, 1)),
    layers.MaxPooling3D(pool_size=(2, 2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # Two output classes (label_1 and label_2)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

