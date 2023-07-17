import json
import colorlover as cl
import os
import dash
from dash import dcc, html
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

json_directory = '../../tf_perf_logs'
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
                                            'Testing Files': [tuple(testing_files)]})
                    data_df = pd.concat([data_df, file_df], ignore_index=True)

grouped = data_df.groupby(['Training Files', 'Testing Files'])
min_accuracy = data_df['Accuracy'].min()
max_accuracy = data_df['Accuracy'].max()

for x in permiTrainingSubjects:
    permiTrainingLength.append(len(x[0]))
    if (len(x[0]) == 104):
        x[0].clear()
        x[0].append("all")

# Initializing color sequence for single subjects
colors = []
for r in range(0, 256, 8):
    for g in range(0, 256, 8):
        for b in range(0, 256, 8):
            colors.append((r, g, b))
            if len(colors) >= 110:
                break
        if len(colors) >= 110:
            break
    if len(colors) >= 110:
        break

colors = ['rgb{}'.format(color) for color in colors]

app = dash.Dash(__name__)
plots = []

for group_name, group_data in grouped:
    file_combination = f"{group_name[0]} - {group_name[1]}"
    file_data = group_data['Accuracy']
    file_names = list(group_name[0]) + list(group_name[1])
    hover_text = [f"File: {permiFileNames[i]} #subjects: {permiTrainingLength[i]} <br> training subjects: {permiTrainingSubjects[i][0]} <br> testing subjects: {permiTestingSubjects[i][0]}" for i in range(
        len(permiFileNames))]

    marker_colors = []
    for i in permiTestingSubjects:
        if len(i[0]) > 1: # More than one testing subject
            marker_colors.append('grey')
        else:
            marker_colors.append(colors[permiTestingSubjects.index(i)%len(colors)]) # Assign a color from list and repeat if more than the colors count

    plot = html.Div(
        className='graph-container',
        children=[
            dcc.Graph(
                figure={
                    'data': [
                        {
                            'x': list(range(len(file_data))),
                            'y': file_data,
                            'mode': 'markers',
                            'marker': {
                                'color': marker_colors,  # using marker colors
                                'size': 15,
                                'line': {
                                    'color': 'Black',
                                    'width': 2
                                }
                            },
                            'name': file_combination,
                            'text': hover_text,
                            'hovertemplate': '<b>%{text}</b><br>Accuracy: %{y:.2f}<extra></extra>'
                        }
                    ],
                    'layout': {
                        'title': f'Accuracy for File Combination: {file_combination}',
                        'xaxis': {'title': 'File', 'range': [0, len(file_data) - 1], 'dtick':1},
                        'yaxis': {'title': 'Accuracy', 'range': [min_accuracy, max_accuracy]}
                    }
                }
            )
        ]
    )
    plots.append(plot)

app.layout = html.Div(children=plots)

if __name__ == '__main__':
    app.run_server(debug=True)

