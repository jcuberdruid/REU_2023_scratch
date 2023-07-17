#{'chunkIndex': '1', 'index': '0', 'subject': '3', 'run': '3', 'epoch': '3'}

import csv

pathannotations = "../../../data/datasets/processed/sequences/MM_RLH_T2_annotation.csv"

# Load annotation
annotations = []
with open(pathannotations, "r") as file:
    reader = csv.DictReader(file)
    rows_per_chunk = {}
    for row in reader:
        annotations.append(row)
        chunk_index = row['chunkIndex']
        if chunk_index in rows_per_chunk:
            rows_per_chunk[chunk_index] += 1
        else:
            rows_per_chunk[chunk_index] = 1

# Check number of rows per chunk
for chunk_index, row_count in rows_per_chunk.items():
    if row_count != 15:
        print(f"Chunk {chunk_index} has {row_count} rows.")
    else:   
        print("ok")
