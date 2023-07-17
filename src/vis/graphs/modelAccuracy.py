import json
import matplotlib.pyplot as plt
import os
import hashlib
import pandas as pd
import seaborn as sns
import numpy as np
import glob

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

def get_model_signature(file_path):
    # Open the file and load the JSON
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Convert the JSON to a string
    data_str = json.dumps(data, sort_keys=True)
    # Compute the hash of the string representation of the JSON
    signature = hashlib.md5(data_str.encode()).hexdigest()
    return signature

# Initialize an empty DataFrame to store the data from JSON files
data_df = pd.DataFrame()

for directory in os.listdir(json_directory):
    if directory.startswith("output"):
        for filename in os.listdir(os.path.join(json_directory, directory)):
            if filename.startswith('output'):
                file_path = os.path.join(json_directory, directory, filename)
                with open(file_path) as file:
                    data = json.load(file)
                    accuracy = data['accuracy']*100
                    testing_subjects = data['testing_subjects']
                    training_subjects = data['training_subjects']

                    if len(testing_subjects) > 1:
                        subject = 110
                    else:
                        subject = testing_subjects[0]

                    training_files = get_array_signature_strings(data['training_files'])
                    testing_files = data['testing_files']

                    # Determine group based on the rules
                    if len(testing_subjects) == len(training_subjects):
                        group = "1:1"
                    elif len(testing_subjects) == 1 and len(training_subjects) > 1:
                        group = "1:n"
                    elif len(testing_subjects) > 1 and len(training_subjects) > 1:
                        group = "n:n"
                    else:
                        group = "1:Clustered"

                    # Obtain the model signature
                    model_file_path = glob.glob(os.path.join(json_directory, directory, 'model_*'))[0] # Changed this line to use glob
                    model_signature = get_model_signature(model_file_path)

                    file_df = pd.DataFrame({
                        'Accuracy': accuracy,
                        'Training Files': training_files,
                        'subject': subject,
                        'Testing Files': [tuple(testing_files)],
                        'Group': group,
                        'Model Signature': model_signature  # add this line
                    })
                    data_df = pd.concat([data_df, file_df], ignore_index=True)

# Group data by Model Signature and calculate mean accuracy and standard deviation
grouped_data = data_df.groupby('Model Signature')['Accuracy'].agg(['mean', 'std'])

# Plotting scatter plot as before
plt.figure(figsize=(10,6), tight_layout=True)
ax1 = sns.scatterplot(data=data_df, x='subject', y='Accuracy', hue='Group', palette='Set2', s=60)
ax1.set(xlabel='Subjects', ylabel='Accuracy %')
ax1.legend(title='Testing Subjects : Training Subjects', title_fontsize = 12)
plt.savefig('AccuracyPerSubjectGrouped.png',  dpi=300)
plt.show()

# Plotting bar graph of average accuracy with error bars
plt.figure(figsize=(10,6), tight_layout=True)
ax2 = grouped_data['mean'].plot(kind='bar', yerr=grouped_data['std'], capsize=4)
ax2.set_ylabel('Average Accuracy %')
plt.title('Average Accuracy per Model')
plt.savefig('AverageAccuracyPerModel.png', dpi=300)
plt.show()

