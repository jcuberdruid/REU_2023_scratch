import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

mapping = {'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCZ', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6', 'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'CZ', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6', 'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPZ', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6', 'Fp1.': 'FP1', 'Fpz.': 'FPZ', 'Fp2.': 'FP2', 'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFZ', 'Af4.': 'AF4', 'Af8.': 'AF8', 'F7..': 'F7', 'F5..': 'F5',
           'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'FZ', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8', 'Ft7.': 'FT7', 'Ft8.': 'FT8', 'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7', 'Tp8.': 'TP8', 'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'PZ', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8', 'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POZ', 'Po4.': 'PO4', 'Po8.': 'PO8', 'O1..': 'O1', 'Oz..': 'OZ', 'O2..': 'O2', 'Iz..': 'IZ'}

channels = ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7',
            'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ']


raw_fnames = eegbci.load_data(1, 3, update_path=True)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raws)

# load standard montages
builtin_montages = mne.channels.get_builtin_montages(descriptions=True)

# rename channels to be uppercase and without dots
mne.rename_channels(raw.info, mapping, allow_duplicates=False, verbose=None)

# load standard 10-10 montage (actually the 10-5 montage which is extended 10-10)
montage_1010 = mne.channels.make_standard_montage("standard_1005")

# set standard 10-10 montage to raw data
raw.set_montage(montage_1010, match_case=False)

# fig = raw.plot_sensors(show_names=True)
# plt.show()

# montage = raw.get_montage()

# for x in montage.dig:
# print(x)
# print(montage.ch_names)
# print(raw.layout)

# Load the layout
layout = mne.channels.find_layout(raw.info)

# layout.plot()

# Get the channel positions
pos = layout.pos

positions = []

for index, x in enumerate(pos):
    row = []
    row.append(x[0]*20)
    row.append(x[1]*20)
    positions.append(row)

xs = [x[0] for x in positions]
ys = [x[1] for x in positions]


# plt.scatter(xs, ys)
# plt.show()

def rasterize_points(points, width, height):
    grid = [['0'] * width for _ in range(height)]  # Initialize the grid
    count = 0
    for point in points:
        x, y = point
        # Convert the floating-point coordinates to integer coordinates
        x = int(x)
        y = int(y)

        # Check if the point falls within the grid boundaries

        if 0 <= x < width and 0 <= y < height:
            # Set the corresponding grid cell to 1
            grid[y][x] = channels[count]
        count = count + 1

    return grid


np.set_printoptions(threshold=np.inf)

# positions = np.array(positions)
grid = rasterize_points(positions, 22, 20)

print(type(grid))
print(type(grid[0]))
print(type(grid[0][0]))

padding = 3

'''
for x in grid:
	for element in x:
	    padded_element = "{:{}}".format(element, padding)
	    print(padded_element, end=' ')
	print()
#(20, 22)
'''
array = np.array(grid)

# Define padding size (adjust as needed)
padding = 2

# Get the maximum length of the strings in the array
max_length = max(len(element) for row in array for element in row)

# Define the format string with padding
format_str = '{:<' + str(max_length + padding) + '}'

# Iterate over the rows and print each element with padding
for row in array:
    for element in row:
        print(format_str.format(element), end='')
    print()  # Move to the next line after each row

array = np.delete(array, 21, axis=1)
array = np.delete(array, 20, axis=1)
array = np.delete(array, 19, axis=1)
array = np.delete(array, 18, axis=1)
array = np.delete(array, 0, axis=1)
array = np.delete(array, 19, axis=0)
array = np.delete(array, 1, axis=0)
array = np.delete(array, 0, axis=0)

print("arr2")

# Define padding size (adjust as needed)
padding = 2

# Get the maximum length of the strings in the array
max_length = max(len(element) for row in array for element in row)

# Define the format string with padding
format_str = '{:<' + str(max_length + padding) + '}'

# Iterate over the rows and print each element with padding
for row in array:
    for element in row:
        print(format_str.format(element), end='')
    print()  # Move to the next line after each row

print(array.shape)

np.save("channelProjection", array, allow_pickle=True, fix_imports=False)
