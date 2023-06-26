import numpy as np
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


path = "../../data/datasets/sequences/MI_RLH_T1.npy"

npyLoad = np.load(path)
print(npyLoad[:5])


#choose a random subject 

#for that subject split epoches into 90:10 split (integer)

#average the 10% into single epoch 

#cluster the average 10 percent with all other data (mayber KNN)

#for each subject output:
# clustered_subject_x_class_xyz.npy + annotations
# training_90%_subject_x_class_xyz.npy + annotations 

