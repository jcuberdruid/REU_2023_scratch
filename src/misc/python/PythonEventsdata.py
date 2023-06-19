import os
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

##mapping for channel names
mapping = {'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCZ', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6', 'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'CZ', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6', 'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPZ', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6', 'Fp1.': 'FP1', 'Fpz.': 'FPZ', 'Fp2.': 'FP2', 'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFZ', 'Af4.': 'AF4', 'Af8.': 'AF8', 'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'FZ', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8', 'Ft7.': 'FT7', 'Ft8.': 'FT8', 'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7', 'Tp8.': 'TP8', 'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'PZ', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8', 'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POZ', 'Po4.': 'PO4', 'Po8.': 'PO8', 'O1..': 'O1', 'Oz..': 'OZ', 'O2..': 'O2', 'Iz..': 'IZ'}
def epoches(subject, test, raw):
	if test == 3 or test == 7 or test == 11:
                event_dict = {"rest":1, "MM_LH":2, "MM_RH":3}
        elif test == 4 or test == 8 or test == 12:
                event_dict = {"rest":1, "MI_LH":2, "MI_RH":3}
        elif test == 5 or test == 9 or test == 13:
                event_dict = {"rest":1, "MM_Fists":2, "MM_Feet":3}
        elif test == 6 or test == 10 or test == 14:
                event_dict = {"rest":1, "MI_Fists":2, "MI_Feet":3}
        else:
                print("Error in data.py: test/task number not in range 3-14")

        print("about to get events")
        events = mne.events_from_annotations(raw, event_id=None)

        print("about to make epoches")
        epochs = mne.Epochs(raw, events[0], tmin=-0.3, tmax=4.2, event_id=event_dict, preload=True)

        #epochs.plot(n_epochs=2)
        evoked_0 = epochs["MI_RH"].average()
        evoked_1 = epochs["MI_LH"].average()

        dicts={'class1':evoked_0,'class2':evoked_1}
        mne.viz.plot_compare_evokeds(dicts)
        mne.viz.plot_evoked(evoked_0)
        mne.viz.plot_evoked(evoked_1)

        plt.show()
        print(epochs["MI_LH"])
	quit()


def loadEEG(subject, test): 
	print(f"loadEEG: subject {subject}, task: {test}")
#	sys.stdout = open('/dev/null', 'w')
	##load data either locally or download 
	raw_fnames = eegbci.load_data(subject, test, update_path=True)
	raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
	raw = concatenate_raws(raws)

	if test == 3 or test == 7 or test == 11: 
		event_dict = {"rest":1, "MM_LH":2, "MM_RH":3}
	elif test == 4 or test == 8 or test == 12: 
		event_dict = {"rest":1, "MI_LH":2, "MI_RH":3}
	elif test == 5 or test == 9 or test == 13: 
		event_dict = {"rest":1, "MM_Fists":2, "MM_Feet":3}
	elif test == 6 or test == 10 or test == 14: 
		event_dict = {"rest":1, "MI_Fists":2, "MI_Feet":3}
	else: 
		print("Error in data.py: test/task number not in range 3-14")

	print("about to get events")
	events = mne.events_from_annotations(raw, event_id=None)
	
	print("about to make epoches")
	epochs = mne.Epochs(raw, events[0], tmin=-0.3, tmax=4.2, event_id=event_dict, preload=True)
	
	#epochs.plot(n_epochs=2)
	evoked_0 = epochs["MI_RH"].average()
	evoked_1 = epochs["MI_LH"].average()
	
	dicts={'class1':evoked_0,'class2':evoked_1}
	mne.viz.plot_compare_evokeds(dicts)
	mne.viz.plot_evoked(evoked_0)
	mne.viz.plot_evoked(evoked_1)
	
	plt.show()
	print(epochs["MI_LH"])
	
	quit()
	print(temp)
	savePath = "./proccessedDB"	
	saveName = 'S'+str(subject)+'_'+str(test)+'.csv'
	np.set_printoptions(suppress=True)
	thisSavePath = os.path.join(savePath, saveName)
	np.savetxt(thisSavePath, temp[0], fmt='%s', delimiter=",", header="lineNum, unknown, T012")
	print(temp)

	return raw 

	##load standard montages
	builtin_montages = mne.channels.get_builtin_montages(descriptions=True)

	##rename channels to be uppercase and without dots
	mne.rename_channels(raw.info, mapping, allow_duplicates=False, verbose=None)

	##load standard 10-10 montage (actually the 10-5 montage which is extended 10-10)
	montage_1010 = mne.channels.make_standard_montage("standard_1020")

	##set standard 10-10 montage to raw data
	raw.set_montage(montage_1010, match_case=False)

	fig = raw.compute_psd(tmax=np.inf, fmax=80).plot(
	    average=True, picks="data", exclude="bads"
	)

	#mne.export.export_raw('stdNoProccess.edf', raw, fmt='auto', physical_range='auto', add_ch_type=False, overwrite=False, verbose=None)

	return raw

