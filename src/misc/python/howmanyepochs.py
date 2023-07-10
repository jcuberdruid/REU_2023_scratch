import mne
from mne.io import concatenate_raws
from mne.datasets import eegbci

def loadEEG(subject, test):
    raw_fnames = eegbci.load_data(subject, test, update_path=True)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)
    # Get events from annotations
    print(raw.annotations)
    quit()
    events = mne.events_from_annotations(raw, event_id=None)

    # Define event dictionary with the desired event labels and IDs
    event_dict = {'event_type1': 1, 'event_type2': 2, 'event_type3': 3}
    # Create epochs based on events and event dictionary
    epochs = mne.Epochs(raw, events[0], tmin=-0.3, tmax=4.2, event_id=event_dict, preload=True)

    # Count the number of epochs and different event types
    num_epochs = len(epochs)
    num_event_types = len(epochs.event_id)
    event_counts = epochs.event_id

    return num_epochs, num_event_types, event_counts

subject = 2  # Replace with the subject number you want to analyze
test = 11  # Replace with the test number you want to analyze
num_epochs, num_event_types, event_counts = loadEEG(subject, test)
print(f"Number of epochs for subject {subject}: {num_epochs}")
print(f"Number of different event types: {num_event_types}")
print("Event counts:")
for event_type, count in event_counts.items():
    print(f"Event type: {event_type} | Count: {count}")

