import json
import os
import dash
from dash import dcc, html
import pandas as pd

# Define the directory where your JSON files are located
json_directory = 'path/to/json/files'

# Initialize an empty DataFrame to store the data from JSON files
data_df = pd.DataFrame()

# Iterate through the JSON files in the directory
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        file_path = os.path.join(json_directory, filename)
        with open(file_path) as file:
            data = json.load(file)
            # Extract the relevant data from the JSON
            accuracy = data['accuracy']
            training_subjects = data['training_subjects']
            
            # Create a new dictionary to store the data
            data_dict = {'Accuracy': accuracy}
            
            # Extract each training subject and add it to the dictionary
            for i, subject in enumerate(training_subjects):
                data_dict[f'Training Subject {i+1}'] = subject
            
            # Append the data to the DataFrame
            data_df = data_df.append(data_dict, ignore_index=True)

# Calculate the correlation between training subjects and accuracy
correlation = data_df.iloc[:, 1:].corrwith(data_df['Accuracy'])

# Sort the correlation values in ascending order
sorted_correlation = correlation.sort_values()

# Determine the maximum subject value
max_subject_value = data_df.iloc[:, 1:].max().max()

# Initialize the Dash application
app = dash.Dash(__name__)

# Create a bar chart to visualize the correlation values
chart = dcc.Graph(
    figure={
        'data': [
            {'x': sorted_correlation.index, 'y': sorted_correlation.values, 'type': 'bar'}
        ],
        'layout': {
            'title': 'Correlation of Training Subjects with Accuracy',
            'xaxis': {'title': 'Training Subjects', 'range': [1, max_subject_value]},
            'yaxis': {'title': 'Correlation', 'dtick': 1}
        }
    }
)

# Create the application layout
app.layout = html.Div(children=[chart])

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

