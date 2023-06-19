#!/bin/bash

find_files() {
    local dir="$1"
    local output_file="$2"

    # Find files with .EDF extension and write their paths to the output file
    find "$dir" -type f -name "*.edf" > "$output_file"
}

# Check if a directory is provided as the first argument
if [ $# -lt 1 ]; then
    echo "Please provide a directory path as the first argument."
    exit 1
fi

# Store the directory path
directory="$1"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "The specified directory does not exist."
    exit 1
fi

# Check if an output file is provided as the second argument
if [ $# -lt 2 ]; then
    echo "Please provide an output file path as the second argument."
    exit 1
fi

# Store the output file path
output_file="$2"

# Call the function to find files and output their paths
find_files "$directory" "$output_file"

echo "File paths have been written to $output_file."

