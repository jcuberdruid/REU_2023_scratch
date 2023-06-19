#!/bin/bash

# Loop through all files in the current directory
for file in *; do
	wc -l $file 
done

