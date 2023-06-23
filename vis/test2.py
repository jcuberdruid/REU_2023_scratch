import json
import os
import dash
from dash import dcc, html
import pandas as pd

# Define the directory where your JSON files are located
json_directory = './files'

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
            training_files = data['training_files']
            testing_files = data['testing_files']
            # Create a DataFrame with the current data
            file_df = pd.DataFrame({'Accuracy': accuracy,
                                    'Training Files': [tuple(training_files)],
                                    'Testing Files': [tuple(testing_files)]})
            # Concatenate the current data DataFrame with the main DataFrame
            data_df = pd.concat([data_df, file_df], ignore_index=True)

# Group the data by training and testing file combinations
grouped = data_df.groupby(['Training Files', 'Testing Files'])

# Determine the overall range of accuracy values
min_accuracy = data_df['Accuracy'].min()
max_accuracy = data_df['Accuracy'].max()

# Initialize the Dash application
app = dash.Dash(__name__)

# Create a line plot for each file combination
plots = []
for group_name, group_data in grouped:
    file_combination = f"{group_name[0]} - {group_name[1]}"
    file_data = group_data['Accuracy']
    plot = dcc.Graph(
        figure={
            'data': [
                {
                    'x': list(range(len(file_data))),
                    'y': file_data,
                    'mode': 'lines+markers',
                    'name': file_combination
                }
            ],
            'layout': {
                'title': f'Accuracy for File Combination: {file_combination}',
                'xaxis': {'title': 'File', 'range': [0, len(file_data) - 1]},
                'yaxis': {'title': 'Accuracy', 'range': [min_accuracy, max_accuracy]}
            }
        }
    )
    plots.append(plot)

# Create the application layout
app.layout = html.Div(children=plots)

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

