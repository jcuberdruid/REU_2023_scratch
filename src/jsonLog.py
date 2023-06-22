import json
import os
from datetime import datetime
from tensorflow import keras

outputPath = '../tf_perf_logs/'

epochJson = None 
modelJson = None 
outputJson = None 

class JSONLogger(keras.callbacks.Callback):
    def __init__(self, filename):
        super(JSONLogger, self).__init__()
        self.filename = filename
        self.log_data = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        epoch_data = {
            'epoch': epoch,
            'loss': float(logs.get('loss')),
            'accuracy': float(logs.get('accuracy')),
            # Add more metrics as needed
        }
        self.log_data.append(epoch_data)
    
    def on_train_end(self, logs=None):
        global epochJson
        epochJson = self.log_data

def output_log(training_subjects, testing_subjects, training_files, testing_files, accuracy):
    # Create a dictionary to hold the data
    data = {
        "training_subjects": training_subjects,
        "testing_subjects": testing_subjects,
        "training_files": training_files,
        "testing_files": testing_files,
        "accuracy": accuracy
    }

    global outputJson 
    outputJson = data


def model_log(model):
    global modelJson 
    modelJson = model

def write_existing_json(data, output_path, output_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{output_name}_{timestamp}.json"
    file_path = output_path + "/" + file_name
    with open(file_path, "w") as outfile:
        outfile.write(data)

def write_log(data, output_path, output_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{output_name}_{timestamp}.json"
    file_path = output_path + "/" + file_name
    with open(file_path, "w") as outfile:
         json.dump(data, outfile)

def make_logs():
    global outputJson
    global modelJson
    global epochJson
    global outputPath 

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    outputDir = f"output_{timestamp}"
    this_output_path = outputPath + outputDir
    os.mkdir(this_output_path)
    #make directory 
    #check that logs exist <- if not generate incomplete 

    #make json logs at directory 
    write_log(epochJson, this_output_path, 'epoch_log')
    write_existing_json(modelJson, this_output_path, 'model_log')
    write_log(outputJson, this_output_path, 'output_log')

'''
# Example usage
input_files = ['epoch_performance.json', 'model.json']
output_file = 'merged.json'
merge_json_files(input_files, output_file)
'''

