import json
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import random

# Define the directory where your JSON files are located
json_directory = '../../tf_perf_logs'

# Initialize an empty DataFrame to store the data from JSON files
data_df = pd.DataFrame()
permiFileNames = []
permiTrainingSubjects = []
permiTrainingLength = []
permiTestingSubjects = []

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
                    accuracy = data['accuracy']
                    training_files = data['training_files']
                    testing_files = data['testing_files']
                    file_df = pd.DataFrame({'Accuracy': accuracy,
                                            'Training Files': [tuple(training_files)],
                                            'Testing Files': [tuple(testing_files)],
                                            'Testing Subject': permiTestingSubjects[-1],  # Added this line
                                            'Filename': permiFileNames[-1]  # Added this line
                                            })
                    data_df = pd.concat([data_df, file_df], ignore_index=True)

# Group the data by training and testing file combinations
grouped = data_df.groupby(['Training Files', 'Testing Files'])

# Initialize the Dash application
app = dash.Dash(__name__)

# Define a function to generate a random color
def generate_random_color():
    return 'rgb(%d, %d, %d)' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Create a scatter plot for each file combination
plots = []
for group_name, group_data in grouped:
    file_combination = f"{group_name[0]} - {group_name[1]}"
    file_data = group_data['Accuracy']
    file_names = group_data['Filename']
    hover_text = [f"File: {file_names[i]} #subjects: {permiTrainingLength[i]} <br> training subjects: {permiTrainingSubjects[i][0]} <br> testing subjects: {permiTestingSubjects[i][0]}" for i in range(len(file_names))]
    plot = html.Div(
        className='graph-container',
        children=[
            dcc.Graph(
                id=f'graph-{file_combination}',  # Give each graph a unique ID
                figure={
                    'data': [
                        {
                            'x': list(range(len(file_data))),
                            'y': file_data,
                            'mode': 'markers',
                            'marker': {
                                'color': generate_random_color(),
                                'size': 10
                            },
                            'name': file_combination,
                            'text': hover_text,
                            'hovertemplate': '<b>%{text}</b><br>Accuracy: %{y:.2f}<extra></extra>'
                        }
                    ],
                    'layout': {
                        'title': f'Accuracy for File Combination: {file_combination}',
                        'xaxis': {'title': 'File', 'range': [0, len(file_data) - 1], 'dtick': 1},
                        'yaxis': {'title': 'Accuracy'}
                    }
                }
            )
        ]
    )
    plots.append(plot)

app.layout = html.Div(children=plots)

@app.callback(
    Output(component_id='graph-all', 'figure'),
    [Input(component_id='graph-all', 'clickData')],
    [State(component_id='graph-all', 'figure')]
)
def update_graph_on_click(clickData, figure):
    if clickData:
        clicked_testing_subject = clickData['points'][0]['customdata'][0]

        for trace in figure['data']:
            if trace['customdata'][0] == clicked_testing_subject:
                trace['marker']['color'] = 'red'
            else:
                trace['marker']['color'] = generate_random_color()

    return figure

if __name__ == '__main__':
    app.run_server(debug=True)

