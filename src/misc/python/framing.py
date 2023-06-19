import numpy as np


def nicePrintNP(arr):
    # Define padding size (adjust as needed)
    padding = 2

    # Get the maximum length of the strings in the arr
    max_length = max(len(element) for row in arr for element in row)

    # Define the format string with padding
    format_str = '{:<' + str(max_length + padding) + '}'

    # Iterate over the rows and print each element with padding
    for row in arr:
        for element in row:
            print(format_str.format(element), end='')
        print()  # Move to the next line after each row


file = "channelProjection.npy"
array = np.load(file)

nicePrintNP(array)
