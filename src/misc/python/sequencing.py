import numpy as np

#read csv 
#each line becomes an image 
#a sequence has 80 of these images (approx 80 <- 0.5 seconds) 

#(num_sequences, num_frames, height, width)
sequence1 = np.random.rand(80, 10, 10)
sequence2 = np.random.rand(80, 10, 10)

#print(sequence_array[0][0])  # <- print single frame


# Create a 4D array
sequence_array = np.array([sequence1, sequence2])

print(sequence_array.shape)  # (2, 80, 100, 100)
print(sequence_array[0][0])  # (2, 80, 100, 100)

#function: line to image 
