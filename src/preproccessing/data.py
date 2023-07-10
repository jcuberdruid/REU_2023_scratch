import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
import pandas as pd
import paths


# mapping for channel names
mapping = {'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCZ', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6', 'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'CZ', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6', 'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPZ', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6', 'Fp1.': 'FP1', 'Fpz.': 'FPZ', 'Fp2.': 'FP2', 'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFZ', 'Af4.': 'AF4', 'Af8.': 'AF8', 'F7..': 'F7', 'F5..': 'F5',
           'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'FZ', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8', 'Ft7.': 'FT7', 'Ft8.': 'FT8', 'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7', 'Tp8.': 'TP8', 'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'PZ', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8', 'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POZ', 'Po4.': 'PO4', 'Po8.': 'PO8', 'O1..': 'O1', 'Oz..': 'OZ', 'O2..': 'O2', 'Iz..': 'IZ'}

def preproccess(subject, test, raw):
    # due to very intentional experimentation best results achieved with (0.1, 0.2) and no h_freq
    raw_filtered = raw.copy().filter(l_freq=1, h_freq=79)  # , verbose=None)
    
    # set up and fit the ICA
    ica = mne.preprocessing.ICA(n_components=32, random_state=None, max_iter='auto')  # , verbose=None)
    ica.fit(raw_filtered)  # , verbose=None)
    
    # Find components to exclude based on correlation with EOG
    eog_inds, scores = ica.find_bads_eog(raw_filtered)
    ica.exclude = eog_inds
    
    # Apply the ICA to the raw data
    raw_corrected = ica.apply(raw_filtered.copy())  # , verbose=None)

    # due to very intentional experimentation best reults achieved with (0.1, 0.2) and no h_freq
    for cutoff in (0.1, 0.2):  # 0.1Hz 0.2Hz
        raw_corrected = raw_corrected.filter(l_freq=cutoff, h_freq=30)  # , verbose=None)
    
    # average referencing
    raw_corrected.set_eeg_reference(ref_channels="average")

    return raw_corrected

def epoches(subject, test, raw):
    print(f"subject {subject}")
    epochSaveDir = paths.dirBase + "trials/" 
    if (os.path.exists(epochSaveDir) != True):
        os.mkdir(epochSaveDir)
    if test == 3 or test == 7 or test == 11:
        event_dict = {"rest": 1, "MM_LH": 2, "MM_RH": 3}
        T1Cond = "MM_LH"
        T2Cond = "MM_RH"
    elif test == 4 or test == 8 or test == 12:
        event_dict = {"rest": 1, "MI_LH": 2, "MI_RH": 3}
        T1Cond = "MI_LH"
        T2Cond = "MI_RH"
    elif test == 5 or test == 9 or test == 13:
        event_dict = {"rest": 1, "MM_Fists": 2, "MM_Feet": 3}
        T1Cond = "MM_Fists"
        T2Cond = "MM_Feet"
    elif test == 6 or test == 10 or test == 14:
        event_dict = {"rest": 1, "MI_Fists": 2, "MI_Feet": 3}
        T1Cond = "MI_Fists"
        T2Cond = "MI_Feet"
    else:
        print("Error in data.py: test/task number not in range 3-14")
    print(f"task is: {test}")
    print("about to get events")
    events = mne.events_from_annotations(raw, event_id=None)
    print(f"events length {len(events)}")
    print(events)
    print("about to make epoches")
    baseline_interval = (-1, 0.0)  # Specify the baseline interval (e.g., 200 ms before the event)
    epochs = mne.Epochs(raw, events[0], tmin=-1, tmax=4.0,
                    event_id=event_dict, reject=None, baseline=baseline_interval, preload=True)

    epochs.crop(tmin=0.0)  # Crop epochs to start at 0.0 seconds (remove the baseline period)

    # Get the drop log from Epochs
    drop_log = epochs.drop_log

    # Iterate over the drop log and print dropped epochs
    print(epochs.events)
    print(drop_log) 
    
    df = epochs.to_data_frame()
    # print(df['condition'].to_string(index=False))
    # df1 = (df.groupby(df['condition'].to_string(index=False) == 'MI_LF')
    # df1, df2 = [x for _, x in df.groupby(df['condition'] == 'MI_LH')]
    #rest, df1 = [x for _, x in df.groupby(df['condition'] == T1Cond)]
    #rest, df2 = [x for _, x in rest.groupby(rest['condition'] == T2Cond)]
    df1 = df[df['condition'] == T1Cond]
    df2 = df[df['condition'] == T2Cond]
    print(len(df1))
    print(len(df2))
    print(df1)
    saveNameDf1 = 'S'+str(subject)+'_'+str(test)+'_T1.csv'
    saveNameDf2 = 'S'+str(subject)+'_'+str(test)+'_T2.csv'
    mne.export.export_raw('stdNoProccessS1_3.edf', raw, fmt='auto', physical_range='auto', add_ch_type=False, overwrite=True, verbose=None)
    df1.insert(1, 'run', test)
    df1.insert(1, 'subject', subject)
    df2.insert(1, 'run', test)
    df2.insert(1, 'subject', subject)

    #df1['run'] = test

    df1.to_csv(epochSaveDir + "/"+saveNameDf1, index=False)
    df2.to_csv(epochSaveDir + "/"+saveNameDf2, index=False)


def loadEEG(subject, test):

    # sys.stdout = open('/dev/null', 'w')
    # load data either locally or download
    raw_fnames = eegbci.load_data(subject, test, update_path=True)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

    # load standard montages
    builtin_montages = mne.channels.get_builtin_montages(descriptions=True)

    # rename channels to be uppercase and without dots
    mne.rename_channels(raw.info, mapping,
                        allow_duplicates=False, verbose=None)

    # load standard 10-10 montage (actually the 10-5 montage which is extended 10-10)
    montage_1010 = mne.channels.make_standard_montage("standard_1020")

    # set standard 10-10 montage to raw data
    raw.set_montage(montage_1010, match_case=False)

    fig = raw.compute_psd(tmax=np.inf, fmax=80).plot(
        average=True, picks="data", exclude="bads"
    )

    # mne.export.export_raw('stdNoProccessS1_3.edf', raw, fmt='auto', physical_range='auto', add_ch_type=False, overwrite=True, verbose=None)

    return raw
