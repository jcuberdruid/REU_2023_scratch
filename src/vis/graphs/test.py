import json
import matplotlib as plt
import os
import dash

import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

from dash import dcc, html
import pandas as pd
# Define the directory where your JSON files are located
json_directory = '../../../tf_perf_logs'

def get_array_signature_strings(arr):
    # Sort the array to ensure consistent order
    sorted_arr = sorted(arr)
    # Convert the sorted array to a string
    arr_str = ','.join(sorted_arr)
    # Compute the hash of the array string
    signature = hashlib.md5(arr_str.encode()).hexdigest()
    return signature


def get_array_signature(arr):
    # Sort the array to ensure consistent order
    sorted_arr = sorted(arr)
    
    # Convert the sorted array to a string
    arr_str = ','.join(map(str, sorted_arr))
    
    # Compute the hash of the array string
    signature = hashlib.md5(arr_str.encode()).hexdigest()
    
    return signature


# Initialize an empty DataFrame to store the data from JSON files
data_df = pd.DataFrame()
# Permi File Names
permiFileNames = []
permiTrainingSubjects = []
permiTrainingLength = []
permiTestingSubjects = []

# /XXX_output/output_log_20230623154717.json


for directory in os.listdir(json_directory):
    if directory.startswith("output"):
        for filename in os.listdir(os.path.join(json_directory, directory)):
            if filename.startswith('output'):
                file_path = os.path.join(json_directory, directory, filename)
                permiFileNames.append(file_path)
                with open(file_path) as file:
                    data = json.load(file)
                    permiTrainingSubjects.append([data['training_subjects']])
                    permiTestingSubjects.append([data['testing_subjects']])
                    accuracy = data['accuracy']*100
                    subjectsArr = data['testing_subjects']
                    if (len(data['training_subjects']) == 0):
                    	print("clustered")

                    if (len(subjectsArr) > 1):
                    	subject = 110
                    else:
                    	subject = subjectsArr[0]
                    training_files = get_array_signature_strings(data['training_files'])
                    testing_files = data['testing_files']
                    file_df = pd.DataFrame({'Accuracy': accuracy, 
                                            'Training Files': training_files,
                                            'subject': subject, 
                                            'Testing Files': [tuple(testing_files)]})
                    data_df = pd.concat([data_df, file_df], ignore_index=True)

for x in data_df['Accuracy']:
	print(x)

print("###")
print(len(data_df['Accuracy']))

plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data=data_df, x='subject', y='Accuracy',  palette='Set2', s=60)
ax.set(xlabel='Subjects', ylabel='Accuracy %')
ax.legend(title='Accuracy per Subject', title_fontsize = 12) 
#ax.set_xticklabels([])
plt.savefig('AccuracyPerSubject.png', dpi=300)
plt.show()

plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data=data_df, x='Training Files', y='Accuracy',  palette='Set2', s=60)
ax.set(xlabel='Unique Class Combinations', ylabel='Accuracy %')
ax.legend(title='Accuracy per Training Data', title_fontsize = 12) 
ax.set_xticklabels([])
plt.savefig('AccuracyTraining.png',  dpi=300)
plt.show()




#read in json 
# care about: accuracy 


#separate into clustered vs not (clustered has empty training subjects)



