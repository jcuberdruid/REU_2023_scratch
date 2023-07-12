import numpy as np

# List of file paths
pathNPY = "../../../data/datasets/processed/sequences/"
#file_paths = ['MM_FF_T1.npy', 'MM_RLH_T1.npy', 'MI_FF_T1.npy', 'MI_FF_T2.npy', 'MI_RLH_T2.npy']  # Add your file paths here
file_paths = ['MM_FF_T2.npy']
# Reshape each file
for file_path in file_paths:
    array = np.load(pathNPY + file_path)
    if array.shape[1] == 80:
        break
    print("###")
    print(array.shape)
    first_axis_size = array.shape[0]
    num_sections = first_axis_size // 80
    reshaped_array = array.reshape((num_sections, 80, *array.shape[1:]))
    print(reshaped_array.shape)
    np.save(pathNPY+file_path, reshaped_array)
