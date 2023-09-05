import numpy as np
import matplotlib.pyplot as plt
import os

def generate_fft_images(data, output_directory):
    if len(data) != 641:
        print("Error: Input data should have length 641")
        return
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # List of electrodes to consider
    electrodes = ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                  'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 
                  'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 
                  'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                  'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ']
    
    # Initialize 2D array with shape (64, 641)
    eeg_data = np.zeros((64, 641))
    
    # Extract EEG data into 2D array
    for i, elec in enumerate(electrodes):
        eeg_data[i] = [float(d[elec]) for d in data]
    
    # Excluding the first index to make it 640 (64 x 10)
    eeg_data = eeg_data[:, 1:]
    
    subject = data[0]['subject']
    run = data[0]['run']
    epoch = data[0]['epoch']
    condition = data[0]['condition']
    
    # Process in sections of 64x64
    for i in range(10):
        segment = eeg_data[:, i*64 : (i+1)*64]
        
        # Compute 2D FFT
        fft_data = np.fft.fft2(segment)
        
        # Convert to magnitude and logarithmically scale
        fft_data_magnitude = np.abs(fft_data)
        fft_data_log = np.log(fft_data_magnitude + 1)
        
        # Create image filename
        filename = f"S{subject}_r{run}_e{epoch}_{condition}_{i+1}.png"
        
        # Save as a grayscale image
        plt.imshow(fft_data_log, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_directory, filename))
        plt.close()

# Example usage:
# generate_fft_images(data, "./output_images")

