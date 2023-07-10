import numpy as np 

path = '/home/jc/keras/data/datasets/hilowonly/sequences/MI_RLH_T1.npy'


data = np.load(path)

print(data.shape)
