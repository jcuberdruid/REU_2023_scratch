import csv

def count_unique_values(csv_file):
    unique_chunk_indexes = set()
    unique_epochs = set()

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            unique_chunk_indexes.add(row['chunkIndex'])
            unique_epochs.add(row['epoch'])

    return len(unique_chunk_indexes), len(unique_epochs)

csv_file = '../../data/datasets/sequences/MI_RLH_T1_annotation.csv'
chunk_indexes, epochs = count_unique_values(csv_file)
print(f"Number of unique chunkIndexes: {chunk_indexes}")
print(f"Number of unique Epochs: {epochs}")

