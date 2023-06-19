import pyedflib
import mne
import os
import numpy as np
import scipy
import pandas as pd
from numpy import asarray
from numpy import savetxt

#processedDb/S97_6.edf
def name(fp):
	fp = fp.split('/')
	print(fp)
	print(fp[1])
	name = fp[1]
	name = name.split('.')
	name = name[0]+".csv"
	return name
'''
def EDFToCSV(path):
	print(os.getcwd())
	f = pyedflib.EdfReader(os.getcwd()+"/"+path)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	data = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
		data[i, :] = f.readSignal(i)
	data = data.transpose()
	np.savetxt("dataCSV/"+name(path), data, delimiter=",")
'''
def EDFToCSV(path):
	edf = mne.io.read_raw_edf(path)
	header = ','.join(edf.ch_names)
	np.savetxt("dataCSV/"+name(path), edf.get_data().T, delimiter=',', header=header)

print(os.getcwd())

fp = open("ProcPaths.txt", 'r')
paths = fp.readlines()
 
for line in paths:
	print(line)
	print(os.getcwd())
	EDFToCSV(line.strip('\n\r'))
