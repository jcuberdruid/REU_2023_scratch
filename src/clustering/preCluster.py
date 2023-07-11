import json
import csv
import os
import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import subjectClusterJson as scj
from datetime import datetime
import time

dataset = "processed4" #NOTE change this for different preproccessing
dataPath = f"../../data/datasets/{dataset}/sequences/"

annotationsPath = ["MI_FF_T1_annotation.csv"]#,"MI_FF_T2_annotation.csv","MI_RLH_T1_annotation.csv","MI_RLH_T2_annotation.csv"]#,"MM_FF_T1_annotation.csv","MM_FF_T2_annotation.csv","MM_RLH_T1_annotation.csv","MM_RLH_T2_annotation.csv"]
dataFiles = ["MI_FF_T1.npy"]#,"MI_FF_T2.npy","MI_RLH_T1.npy","MI_RLH_T2.npy"]#,"MM_FF_T1.npy","MM_FF_T2.npy","MM_RLH_T1.npy","MM_RLH_T2.npy"]

# for padding
fixed_column = 50

#Load annotations
annotations = []
for x in annotationsPath:
    loading_str = f"loading {x}"
    print(loading_str, end='')
    padding = ' ' * (fixed_column - len(loading_str))
    annotations_arr = []
    path = (dataPath + x)
    with open(path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            annotations_arr.append(row)
    annotations.append(annotations_arr)
    print(f"{padding}done.")

#Load data 
data = []
for x in dataFiles:
    loading_str = f"loading {x}"
    time.sleep(0.5)
    print(loading_str, end='')
    padding = ' ' * (fixed_column - len(loading_str))
    path = (dataPath+x)
    data.append(np.load(path))
    print(f"{padding}done.")
print("annotations")
print(annotations[0][0])
print("data")
print(data[0][0])
