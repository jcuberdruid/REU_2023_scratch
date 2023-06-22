import json
from tensorflow import keras

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
        with open(self.filename, 'w') as json_file:
            json.dump(self.log_data, json_file)


json_logger = JSONLogger('epoch_performance.json')
model.fit(x_train, y_train, epochs=10, callbacks=[json_logger])

