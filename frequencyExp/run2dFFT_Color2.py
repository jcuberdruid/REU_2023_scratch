import numpy as np
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

#local 
import paths

classesPath = paths.dirBase+"classes/"
outputDir = paths.dirBase+"ffts/"

def generate_fft_images(data, output_directory):
    if len(data) != 641:
        print("Error: Input data should have length 641")
        return
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    sample_rate = 160  # Your sample rate is 160 samples per second
    fft_size = 64  # The size of your FFT is 64
    delta_f = sample_rate / fft_size  # Frequency resolution

    # Define frequency band ranges in terms of indices
    alpha_range = slice(int(8/delta_f), int(13/delta_f))
    beta_range = slice(int(13/delta_f), int(30/delta_f))
    gamma_range = slice(int(30/delta_f), int(100/delta_f))
    
    # ... (rest of the code remains the same)
    
    for i in range(10):
        segment = eeg_data[:, i*64 : (i+1)*64]
        fft_data = np.fft.fft2(segment)
        fft_data_magnitude = np.abs(fft_data)
        fft_data_log = np.log(fft_data_magnitude + 1)
        
        # Apply color mapping for alpha, beta, gamma ranges
        colored_fft_data_log = np.stack([fft_data_log]*3, axis=2)
        colored_fft_data_log[alpha_range, :, 0] = 255  # Red for alpha
        colored_fft_data_log[beta_range, :, 1] = 255   # Green for beta
        colored_fft_data_log[gamma_range, :, 2] = 255  # Blue for gamma
        
        filename = f"S{subject}_r{run}_e{epoch}_{condition}_{i+1}.png"
        
        plt.imshow(colored_fft_data_log)
        plt.axis('off')
        plt.savefig(os.path.join(output_directory, filename), bbox_inches='tight', pad_inches=0)
        plt.close()

def group_rows_preserve_order(rows, keys):
    groups = []
    current_group = []
    prev_key_values = None
    for row in rows:
        key_values = [row[key] for key in keys]
        if key_values != prev_key_values and current_group:
            groups.append(current_group)
            current_group = []
        current_group.append(row)
        prev_key_values = key_values
    if current_group:
        groups.append(current_group)
    return groups

def chunk_each(csv_file):
    csv_file = classesPath + csv_file
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
        key_columns = ['subject', 'epoch', 'run']
        row_groups = group_rows_preserve_order(rows, key_columns)
        for chunk in row_groups:
            generate_fft_images(chunk, outputDir)
            exit(0)

def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            csv_files.append(filename)
    return csv_files

csv_files_list = get_csv_files(classesPath)

for x in csv_files_list:
    print(f"proccessing file {x}")
    chunk_each(x)
