import numpy as np
import pandas as pd

#NOTE the current implementation takes 9.2 seconds so around 6hrs for whole dataset - not counting storage - considering testing c/cpp 

# Define the headers you want to use
headers_to_use = ["FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
                  "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
                  "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
                  "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8",
                  "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
                  "FT7", "FT8", "T7", "T8", "T9", "T10",
                  "TP7", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
                  "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ"]

df = pd.read_csv("../data/datasets/unprocessed/classes/MM_RLH_T2.csv") #NOTE seperate reads of large file is inefficent, consider putting in worker task manager #but multiplex so 
unique_combination = {'subject': 72, 'run': 3, 'epoch': 1}
filtered_df = df.loc[(df['subject'] == unique_combination['subject']) &
                     (df['run'] == unique_combination['run']) &
                     (df['epoch'] == unique_combination['epoch'])]

eeg_channel_image = filtered_df[headers_to_use].values.tolist() #NOTE maybe input to this file function 


#TODO symmetry in CNN input data is desired but 4 seconds at 160 sps is already limited (don't know enough need to look into effects on transform) 

def perform_2d_fourier_transform(image_641x64):
    if not all(len(row) == 64 for row in image_641x64) or len(image_641x64) != 641: #NOTE how to split 4 second interval 
        raise ValueError("Input should be a 641x64 array.") #XXX see above 
    image_array = np.array(image_641x64)
    fourier_transform = np.fft.fft2(image_array)
    fourier_transform_shifted = np.fft.fftshift(fourier_transform)
    magnitude_spectrum = np.abs(fourier_transform_shifted)
    return magnitude_spectrum

sample_image = eeg_channel_image
result = perform_2d_fourier_transform(sample_image)

#TODO make .npy datafile for result -inefficent but meh 
