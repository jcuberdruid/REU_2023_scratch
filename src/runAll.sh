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

for ((i=40; i<=109; i++))
do
    if [[ $i -eq 88 || $i -eq 89 || $i -eq 92 || $i -eq 100 || $i -eq 104 ]]; then
	continue
    fi
    python3 main.py -test $i
done
