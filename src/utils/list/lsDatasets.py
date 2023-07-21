import os

# specify the directory you want to list files from
dir_path = '../../../data/datasets/'

files = os.listdir(dir_path)

for file in files:
    print(file)


