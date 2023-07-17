import pandas as pd

file_path =  '../../data/datasets/processed4/sequences/MM_RLH_T1_annotation.csv'
# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Count the number of unique values in the "chunks" column
unique_chunks_count = data["chunkIndex"].nunique()

# Print the result
print("Number of unique chunks:", unique_chunks_count)

