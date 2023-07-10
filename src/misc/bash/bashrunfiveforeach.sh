#!/bin/bash

venv_path="./venv/bin/activate"
# Check if the virtual environment exists
if [[ -f $venv_path ]]; then
  # Activate the virtual environment
  . "$venv_path"
  echo "Virtual environment activated!"
else
  echo "Virtual environment not found!"
fi

for ((j=1; j<=109; j++))
do
	for ((i=$j+1; i<=109; i++))
	do
	    if [[ $i -eq 88 || $i -eq 89 || $i -eq 92 || $i -eq 100 || $i -eq 104 ]]; then
		continue
	    fi

	    # Your code here. Replace the echo statement with your desired commands.
	    python3 runfiveforEach.py $j $i
	done
done

