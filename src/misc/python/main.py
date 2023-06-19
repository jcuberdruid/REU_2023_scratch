
## 0<->63 electrodes, units: 

from matplotlib import pyplot as plt 
import pyedflib
import numpy as np
import scipy
import pandas as pd
from numpy import asarray
from numpy import savetxt

file_name = "eegmmidb/1.0.0/S100/S100R01.edf"

f = pyedflib.EdfReader(file_name)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
data = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
        data[i, :] = f.readSignal(i)


