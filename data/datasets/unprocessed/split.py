import os
import shutil 

file_path = "./fft_grey/"
file_patterns = ["MI_Fists","MI_Feet","MI_LH","MI_RH","MM_Fists","MM_Feet","MM_LH","MM_RH"]

for x in file_patterns:
    if not os.path.exists(os.path.join(file_path, x)):
        os.mkdir(os.path.join(file_path, x))

data_files = os.listdir(file_path)

for x in data_files: 
    for y in file_patterns:
        if y not in x:
            continue
        if not os.path.isdir(os.path.join(file_path, x)):
            print(f"moving {x} to {y}")
            shutil.move(os.path.join(file_path, x), os.path.join(file_path, y, x))
