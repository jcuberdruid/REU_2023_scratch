#!/bin/bash

for ((i=1; i<=109; i++))
do
  padded_number=$(printf "%03d" $i)
  prefix="S$padded_number"
  files=$(ls -1 "$prefix"* 2>/dev/null)
  count=$(echo "$files" | wc -l)
  echo "$prefix: $count"
done

