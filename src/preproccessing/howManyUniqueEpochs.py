import csv

def count_unique_runs_epochs(csv_file):
    unique_runs_epochs = {}

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            subject = row['subject']
            run = row['run']
            epoch = row['epoch']
            run_epoch = (run, epoch)

            if subject not in unique_runs_epochs:
                unique_runs_epochs[subject] = set()

            unique_runs_epochs[subject].add(run_epoch)

    return unique_runs_epochs

# Example usage
#csv_file_path = '/home/jc/keras/data/datasets/unproccessed/tasks/S41_MM_RLH_T1.csv'
#csv_file_path = '/home/jc/keras/data/datasets/unproccessed/trials/csvs/S41_3_T1.csv'
#csv_file_path = '/home/jc/keras/data/datasets/unproccessed/trials/csvs/S41_7_T1.csv'
csv_file_path = '/home/jc/keras/data/datasets/unproccessed/trials/csvs/S41_11_T1.csv'


unique_runs_epochs = count_unique_runs_epochs(csv_file_path)
# Print the results
for subject, runs_epochs in unique_runs_epochs.items():
    print(f"Subject {subject}:")
    print(f"Number of unique runs and epochs combinations: {len(runs_epochs)}")
    print("---")

