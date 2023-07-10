import mne

subject_ids = list(range(60, 80))  # Download all subjects (1-109)
runs = list(range(1, 15))  # Download all runs (1-14)

for subject in subject_ids:
    for run in runs:
        print(f"Downloading Subject {subject}, Run {run}")
        mne.datasets.eegbci.load_data(subject, run, update_path=True)

