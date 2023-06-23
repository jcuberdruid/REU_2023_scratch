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
            training_subjects = data['training_subjects']
            # Append the data to the DataFrame
            data_df = pd.concat([data_df, pd.DataFrame({'Accuracy': [accuracy], 'Training Subjects': [training_subjects]})])

# Calculate the correlation between training subjects and accuracy
correlation = data_df['Training Subjects'].apply(pd.Series).corrwith(data_df['Accuracy'])

# Sort the correlation values in ascending order
sorted_correlation = correlation.sort_values()

# Determine the maximum subject value
max_subject_value = max(data_df['Training Subjects'].apply(lambda x: max(x)))

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
            'xaxis': {'title': 'Training Subjects', 'range': [1, max_subject_value], 'dtick': 1},
            'yaxis': {'title': 'Correlation', 'dtick': 1}
        }
    }
)

# Create the application layout
app.layout = html.Div(children=[chart])

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

