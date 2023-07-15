import csv
import json
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

clusteringPath = '../../logs/clustering_logs/'

dataset = "processed7" #NOTE change this for different preproccessing 
dataPath = f"../../data/datasets/{dataset}/sequences/"

annotationsPath = ["MI_FF_T1_annotation.csv","MI_FF_T2_annotation.csv","MI_RLH_T1_annotation.csv","MI_RLH_T2_annotation.csv"]#,"MM_FF_T1_annotation.csv","MM_FF_T2_annotation.csv","MM_RLH_T1_annotation.csv","MM_RLH_T2_annotation.csv"]
dataFiles = ["MI_FF_T1.npy","MI_FF_T2.npy","MI_RLH_T1.npy","MI_RLH_T2.npy"]#,"MM_FF_T1.npy","MM_FF_T2.npy","MM_RLH_T1.npy","MM_RLH_T2.npy"]

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


def cluster(targetSubject, subjectObject, video_data, data_npy, annotations_arr, start_clusters=15):
    subject = subjectObject #TODO go back through and change subject variable name manually 
    print("###############START OF Class#############")
    # constants
    #targetSubject = 24 
    #annotations_csv = "../../data/datasets/hilowonly/sequences/MI_RLH_T1_annotation.csv"
    #data_npy = "../../data/datasets/hilowonly/sequences/MI_RLH_T1.npy"

    ##############################################
    # 90-10 split
    ##############################################


    # find the indices of the subject in the annotations_arr
    subjectIndices = []
    for x in annotations_arr:
        if int(x['subject']) == targetSubject:
            subjectIndices.append(x)

    # shuffle the indices at random
    random.shuffle(subjectIndices)

    # set the clusteringIndices to 10 percent of the shuffled indices
    clusteringIndices = []
    testingIndices = []

    split_ratio = 0.9
    split_index = int(len(subjectIndices) * split_ratio)

    # Split the array
    testingIndices = subjectIndices[:split_index] # long one 
    clusteringIndices = subjectIndices[split_index:] # short one 
    #print("90% Array:", len(testingIndices))
    #print("10% Array:", len(clusteringIndices))
    # remove all the testingIndices from the sequences pointed at by the subjects annotations from np
    # do the same from the data csv
    #{'chunkIndex': '1926', 'index': '28880', 'subject': '8', 'run': '12', 'epoch': '25'}

    delete_indices = [int(y['index']) for y in subjectIndices]
    target_indices = [int(y['index']) for y in clusteringIndices]
    
    annotations_arr = [i for j, i in enumerate(annotations_arr) if j not in delete_indices]

    print(f"video_data in KNN shape {video_data.shape}")
    ##############################################
    # KNN 
    ##############################################

    # Reshape the video data to 2D vectors
    n_videos, n_frames, width, height = video_data.shape
    video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

    # Standardize the data
    scaler = StandardScaler()
    video_data_2d = scaler.fit_transform(video_data_2d)
    print(f"video_data_2d in KNN shape {video_data_2d.shape}")

    target_videos_2d = video_data_2d[target_indices]
    video_data_2d = np.delete(video_data_2d, delete_indices, axis=0)

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='cosine')
    nbrs.fit(target_videos_2d)

    distances, indices = nbrs.kneighbors(video_data_2d  )

    sorted_indices = np.argsort(distances, axis=0)
    sorted_other_videos = video_data_2d[sorted_indices]

    clusteredEpochs = [] ## contains dicts 
    for index, x in enumerate(annotations_arr):
        thisDict = {'subject': int(x['subject']), 'epoch': int(x['epoch']), 'closeness': sorted_indices[index], 'index':int(x['index'])}
        clusteredEpochs.append(thisDict)


    #str, testingIndices: [], trainingIndices: [], similarIndices: [] = None
    subject = scj.Subject(targetSubject)
    testingIndicesInt = [int(d['index']) for d in testingIndices]
    clusteringIndicesInt = [int(d['index']) for d in clusteringIndices]
    class_n = scj.TrainingClass(data_npy, testingIndicesInt, clusteringIndicesInt)

    #{'subject': 24, 'epoch': 3, 'cluster': 1, 'index': 343}
    thisSubject = 1 
    thisEpochs = []
    thisIndices = []

    #NOTE change for KNN to instead add the lowest indices up to a certain ammount 
    for x in clusteredEpochs:
        if x['closeness'] <= 1000  and x['subject'] != targetSubject:
            if thisSubject != x['subject']:
                class_n.appendSimilar(thisSubject, thisEpochs, thisIndices)
                thisEpochs.clear()    
                thisIndices.clear()    
                thisSubject = x['subject'] 
            thisIndices.append(x['index'])
            if x['epoch'] not in thisEpochs: 
                thisEpochs.append(x['epoch'])
    class_n.appendSimilar(thisSubject, thisEpochs, thisIndices)
    print("###############END OF Class#############")
    return class_n
#NOTE end of func 


#TODO mainloop 
exclude = [88, 89, 92, 100, 104]
subjects = [x for x in range(1, 110) if x not in exclude]

dirPath = clusteringPath + "proccessed7KNN_2000Closest" + "/"
if not os.path.exists(dirPath):
    os.mkdir(dirPath)

for x in subjects:
    print(f"Clustering for subject {x}")
    subject = scj.Subject(x)
    for y in range(len(annotationsPath)):
        class_n = cluster(x, subject, data[y], dataFiles[y], annotations[y])
        subject.appendClasses(class_n)
    json_string = subject.toJson()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fileName = f"S{x}_clustering_log_{timestamp}"
    filePath = dirPath + fileName 
    with open(filePath, 'w') as f:
        f.write(json_string)
