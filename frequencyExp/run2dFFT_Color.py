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
        
    electrodes = ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                  'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ',
                  'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7',
                  'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                  'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ']

    eeg_data = np.zeros((64, 641))
    
    for i, elec in enumerate(electrodes):
        eeg_data[i] = [float(d[elec]) for d in data]

    eeg_data = eeg_data[:, 1:]
    subject = data[0]['subject']
    run = data[0]['run']
    epoch = data[0]['epoch']
    condition = data[0]['condition']

    for i in range(10):
        segment = eeg_data[:, i*64 : (i+1)*64]
        
        # Compute 2D FFT
        fft_data = np.fft.fft2(segment)
        
        # Convert to magnitude and logarithmically scale
        fft_data_magnitude = np.abs(fft_data)
        fft_data_log = np.log(fft_data_magnitude + 1)
        fft_data_normalized = (fft_data_log / fft_data_log.max()) * 255
        
        plt.imshow(fft_data_normalized, cmap='gray')

        # Create semi-transparent overlays for the frequency bands
        alpha_range = slice(3, 6)
        beta_range = slice(6, 13)
        gamma_range = slice(13, 41)
        
        plt.imshow(np.ma.masked_where(fft_data_normalized[alpha_range, alpha_range] == 0, fft_data_normalized[alpha_range, alpha_range]), cmap='Reds', alpha=0.4, extent=(alpha_range.start, alpha_range.stop, alpha_range.start, alpha_range.stop))
        plt.imshow(np.ma.masked_where(fft_data_normalized[beta_range, beta_range] == 0, fft_data_normalized[beta_range, beta_range]), cmap='Greens', alpha=0.4, extent=(beta_range.start, beta_range.stop, beta_range.start, beta_range.stop))
        plt.imshow(np.ma.masked_where(fft_data_normalized[gamma_range, gamma_range] == 0, fft_data_normalized[gamma_range, gamma_range]), cmap='Blues', alpha=0.4, extent=(gamma_range.start, gamma_range.stop, gamma_range.start, gamma_range.stop))

        # Remove axis
        plt.axis('off')
        
        # Remove white borders
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        filename = f"S{subject}_r{run}_e{epoch}_{condition}_{i+1}.png"
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
