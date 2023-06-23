import json
import os
import dash
from dash import dcc, html
import pandas as pd
import numpy as np

# Define the directory where your JSON files are located
json_directory = './files'

# Initialize an empty DataFrame to store the data from JSON files
data_df = pd.DataFrame()

# Iterate through the JSON files in the directory
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        file_path = os.path.join(json_directory, filename)
        with open(file_path) as file:
            print(file_path) 
            data = json.load(file)
            # Extract the relevant data from the JSON
            accuracy = data['accuracy']
            training_files = data['training_files']
            testing_files = data['testing_files']
            # Create a DataFrame with the current data
            file_df = pd.DataFrame({'Accuracy': accuracy,
                                    'Training Files': training_files,
                                    'Testing Files': testing_files})
            # Concatenate the current data DataFrame with the main DataFrame
            data_df = pd.concat([data_df, file_df], ignore_index=True)

# Group the data by training and testing file combinations
grouped = data_df.groupby(['Training Files', 'Testing Files'])
# Assuming you have a DataFrame called 'grouped' obtained after grouping your data
for group_name, group_data in grouped:
    print("Group Name:", group_name)
    print(group_data)
    print("------------------")


# Calculate the average accuracy for each group
average_accuracy = grouped['Accuracy'].mean()

# Calculate the standard error for each group
std_error = grouped['Accuracy'].sem()

# Get unique combinations of training and testing files
file_combinations = data_df.groupby(['Training Files', 'Testing Files']).groups.keys()

# Initialize the Dash application
app = dash.Dash(__name__)

# Create a bar chart to visualize average accuracy with error
chart = dcc.Graph(
    figure={
        'data': [
            {
                'x': list(file_combinations),
                'y': average_accuracy,
                'error_y': dict(type='data', array=std_error, visible=True),
                'type': 'bar'
            }
        ],
        'layout': {
            'title': 'Average Accuracy with Error by File Combinations',
            'xaxis': {'title': 'File Combinations'},
            'yaxis': {'title': 'Accuracy'}
        }
    }
)

# Create the application layout
app.layout = html.Div(children=[chart])

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

