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

clusteringPath = '../../clustering_logs/'

dataset = "hilowonly" #NOTE change this for different preproccessing 
dataPath = f"../../data/datasets/{dataset}/sequences/"

annotationsPath = ["MI_FF_T1_annotation.csv","MI_FF_T2_annotation.csv","MI_RLH_T1_annotation.csv","MI_RLH_T2_annotation.csv","MM_FF_T1_annotation.csv","MM_FF_T2_annotation.csv","MM_RLH_T1_annotation.csv","MM_RLH_T2_annotation.csv"]
dataFiles = ["MI_FF_T1.npy","MI_FF_T2.npy","MI_RLH_T1.npy","MI_RLH_T2.npy","MM_FF_T1.npy","MM_FF_T2.npy","MM_RLH_T1.npy","MM_RLH_T2.npy"]

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


def cluster(targetSubject, subjectObject, video_data, data_npy, annotations_arr):
    subject = subjectObject #TODO go back through and change subject variable name manually 
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

    delete_indices = [int(y['index']) for y in testingIndices]
    annotations_arr = [i for j, i in enumerate(annotations_arr) if j not in delete_indices]
    video_data = np.delete(video_data, delete_indices, axis=0)

    ##############################################
    # SOM
    ##############################################

    # Reshape the video data to 2D vectors
    n_videos, n_frames, width, height = video_data.shape
    video_data_2d = video_data.reshape(n_videos, n_frames * width * height)

    # Create a SOM instance
    som = MiniSom(x=1, y=2, input_len=n_frames * width * height, sigma=0.8, learning_rate=0.5)

    # Initialize the SOM
    som.random_weights_init(video_data_2d)

    # Train the SOM
    som.train_random(video_data_2d, num_iteration=1000)

    # Get the clustered labels for each video
    cluster_labels = np.zeros(n_videos)
    for i in range(n_videos):
        video_vector = video_data_2d[i]
        cluster_labels[i] = som.winner(video_vector)[-1]

    counts = np.bincount((cluster_labels.astype(int)))

    clusteredEpochs = []
    for index, x in enumerate(annotations_arr):
        thisDict = {'subject': int(x['subject']), 'epoch': int(
            x['epoch']), 'cluster': int(cluster_labels[index]), 'index':int(x['index'])}
        clusteredEpochs.append(thisDict)
    targetClusteredIndices = [int(y['index']) for y in clusteringIndices]
    targetClustered = [i for j, i in enumerate(clusteredEpochs) if i['index'] in targetClusteredIndices]

    def calc_percentage_subjects_per_cluster(arr):
        from collections import defaultdict

        # A dictionary to keep track of the unique clusters each subject is part of
        subject_cluster_map = defaultdict(set)

        # Populating the dictionary
        for record in arr:
            subject_cluster_map[record['subject']].add(record['cluster'])

        # A dictionary to keep track of the number of subjects in each number of clusters
        cluster_count_map = defaultdict(int)

        # A dictionary to keep track of which subjects are in each category
        subjects_per_cluster_count = defaultdict(list)

        # Counting the number of subjects in each number of clusters and populating subjects_per_cluster_count
        for subject, clusters in subject_cluster_map.items():
            cluster_count = len(clusters)
            cluster_count_map[cluster_count] += 1
            subjects_per_cluster_count[cluster_count].append(subject)

        total_subjects = len(subject_cluster_map)

        # Calculate and print percentage for each number of clusters, and print the subjects in each category
        for num_clusters, num_subjects in sorted(cluster_count_map.items()):
            percentage = (num_subjects / total_subjects) * 100
        #print(f"{percentage:.2f}% of subjects are spread across {num_clusters} clusters: {subjects_per_cluster_count[num_clusters]}")

    def find_similar_subjects(data):
        subject_clusters = {}
        result = []

        # Calculate the count of indices in each cluster for each subject
        for item in data:
            subject = item['subject']
            cluster = item['cluster']
            if subject not in subject_clusters:
                subject_clusters[subject] = {}
            subject_clusters[subject][cluster] = subject_clusters[subject].get(cluster, 0) + 1

        sorted_subjects = sorted(subject_clusters.keys())
        for subject in sorted_subjects:
            clusters = subject_clusters[subject]
            total_indices = sum(clusters.values())
            percentages = {cluster: count / total_indices * 100 for cluster, count in clusters.items()}
            subject_clusters[subject] = percentages
            #print(f"subject {subject}, percentages: {percentages}")
        # Find subjects with similar distributions
        for subject, percentages in subject_clusters.items():
            similar_subjects = [s for s, p in subject_clusters.items() if p == percentages and s != subject]
            if similar_subjects:
                result.append((subject, similar_subjects))

    calc_percentage_subjects_per_cluster(clusteredEpochs)

    # NOTE for later, for now start with two clusters only
    # for the target subject is 90 percent of the subjects clusteredIndices in one cluster?
    # yes: then make json and define similar subjects etc 
    # no: then decrease cluster size (or are there subjects with a similar distribution?)


    # NOTE will need to change when testing more than two clusters 
    #which cluster does the subject belong to 
    subjectsCluster = 1
    clusterCount0 = 0;
    clusterCount1 = 0;
    for x in targetClustered:
        if x['cluster'] == 0:
            clusterCount0 = clusterCount0 + 1
        else:
            clusterCount1 = clusterCount1 + 1

    if clusterCount0 > clusterCount1: 
        subjectsCluster = 0

    # create json 
    subject = scj.Subject(targetSubject)
    #str, testingIndices: [], trainingIndices: [], similarIndices: [] = None

    testingIndicesInt = [int(d['index']) for d in testingIndices]
    clusteringIndicesInt = [int(d['index']) for d in clusteringIndices]

    class_n = scj.TrainingClass(data_npy, testingIndicesInt, clusteringIndicesInt)

    #{'subject': 24, 'epoch': 3, 'cluster': 1, 'index': 343}
    thisSubject = 1 
    thisEpochs = []
    thisIndices = []

    for x in clusteredEpochs:
        if x['cluster'] == subjectsCluster and x['subject'] != targetSubject:
            if thisSubject != x['subject']:
                class_n.appendSimilar(thisSubject, thisEpochs, thisIndices)
                thisEpochs.clear()    
                thisIndices.clear()    
                thisSubject = x['subject'] 
            thisIndices.append(x['index'])
            if x['epoch'] not in thisEpochs: 
                thisEpochs.append(x['epoch'])

    class_n.appendSimilar(thisSubject, thisEpochs, thisIndices)
    return class_n
#NOTE end of func 


#TODO mainloop 
exclude = [88, 89, 92, 100, 104]
subjects = [x for x in range(1, 110) if x not in exclude]

dirPath = clusteringPath + dataset + "/"
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
