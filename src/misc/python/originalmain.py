import os
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
## MNE proccessing
import mne
#import data
from multiprocessing import Process, freeze_support
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

'''
savePath = "./proccessedDB"
#sample_data_raw_file = "eegmmidb/1.0.0/S036/S036R07.edf"
sample_data_raw_file = "S004R04.edf"
raw = mne.io.read_raw_edf(sample_data_raw_file, stim_channel='auto')
raw.crop(0, 60).load_data()  # just use a fraction of data for speed here
'''
raw_fnames = eegbci.load_data(36, 7, update_path=True)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raws)

temp = mne.events_from_annotations(raw, event_id=None)

np.savetxt("foo.csv", temp[0], delimiter=",", header="T0, T1, T2")
print(temp)
'''
#highpass 
##due to very intentional experimentation best reults achieved with (0.1, 0.2) and no h_freq
for cutoff in (0.1, 0.2): #0.1Hz 0.2Hz 
    raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=30) #, verbose=None)
# set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=30, random_state=97, max_iter=800) #, verbose=None)
ica = ica.fit(raw_highpass)#, verbose=None)
ica = ica.apply(raw_highpass)#, verbose=None)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
#average referencing 
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels="average")
saveName = 'S'+str(subject)+'_'+str(run)+'.edf'
thisSavePath = os.path.join(savePath, saveName)
mne.export.export_raw(thisSavePath, raw_avg_ref, fmt='auto', physical_range='auto', add_ch_type=False, overwrite=True)#, verbose=None)
print("exported subject"+str(subject)+", run: "+str(run))	
'''
