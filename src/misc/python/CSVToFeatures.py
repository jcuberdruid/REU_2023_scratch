from dataclasses import dataclass as dc
from pathlib import Path
import csv
import re
from numpy import savetxt
from numpy import asarray
import pandas as pd
import scipy
import numpy as np
import os
import mne
import pyedflib

headers = []
dirName = "t1t2"
currentLineCnt = 0

eventDir = "DbEvents"
dataDir = "dataCSV"


def lineCnt(file_path):
    with open(file_path, "r") as file:
        return sum(1 for line in file)


def getSubject(string):
    pattern = r".*?(\d{1,3}).*"
    match = re.search(pattern, string)
    if match:
        number = match.group(1)
        return number


def getTask(string):
    pattern = r".*?(\d{1,3})\D*$"
    match = re.search(pattern, string)
    if match:
        number = match.group(1)
        return number


def write_row_to_csv(csv_path, row):
    Path(csv_path).touch()
    with open(csv_path, "a", newline="\n") as file:
        writer = csv.writer(file)
        if os.path.isfile(csv_path) and os.stat(csv_path).st_size == 0:
            writer.writerow(headers)
        writer.writerow(row)


def copy_rows(csv_path, start_row, end_row):
    values = []

    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        for row_num, row in enumerate(reader, start=1):
            if row_num >= start_row and row_num < end_row:
                values.append(row)
                # if row and row[0].isdigit():
                # values.append(int(row[0]))
    return values


def split_csv(data_path, event_path):
    t0Sum = 0
    t1 = []
    t2 = []
    currentRowT = 0
    # use .extend() for merging string lists
    with open(event_path, "r") as file:
        reader = csv.reader(file)
        start_row = 0
        for row_num, row in enumerate(reader, start=1):
            end_row = row[0]
            if row_num == 1 or row_num == 2:
                start_row = row[0]
                currentRowT = row[2]
                continue

            # go through each row in the events csv skipping the header
            if int(currentRowT) == 1:  # if t0 aka resting skip
                t0Sum += int(end_row) - int(start_row)
                start_row = row[0]
                currentRowT = row[2]
                continue
            if int(currentRowT) == 2:
                t1.extend(copy_rows(data_path, int(
                    start_row) + 1, int(end_row) + 1))
                start_row = row[0]
                currentRowT = row[2]
                continue
            if int(currentRowT) == 3:
                t2.extend(copy_rows(data_path, int(
                    start_row) + 1, int(end_row) + 1))
                start_row = row[0]
                currentRowT = row[2]
                continue
    LC = lineCnt(data_path)
    if int(currentRowT) == 1:  # if t0 aka resting skip
        t0Sum += int(end_row) - int(start_row)
    if int(currentRowT) == 2:
        t1.extend(copy_rows(data_path, int(start_row) + 1, LC + 1))
    if int(currentRowT) == 3:
        t2.extend(copy_rows(data_path, int(start_row) + 1, LC + 1))

    t1Path = (
        dirName
        + "/T1_S"
        + str(getSubject(data_path))
        + "_"
        + str(getTask(data_path))
        + ".csv"
    )
    t2Path = (
        dirName
        + "/T2_S"
        + str(getSubject(data_path))
        + "_"
        + str(getTask(data_path))
        + ".csv"
    )

    if t1[0][1] == "FC3":
        headers.extend(t1.pop(0))
        headers.append("subject")
        headers.append("task")
    if t2[0][1] == "FC3":
        headers.extend(t2.pop(0))
        headers.append("subject")
        headers.append("task")

    for x in t1:
        x.append(getTask(data_path))
        x.append(getSubject(data_path))
        write_row_to_csv(t1Path, x)

    for x in t2:
        x.append(getTask(data_path))
        x.append(getSubject(data_path))
        write_row_to_csv(t2Path, x)

    #print("number of t0 lines: " + str(t0Sum))
    #print("original Line count: " + str(LC))


print(os.getcwd())
fp = open("universalPaths.txt", "r")
paths = fp.readlines()
for line in paths:
	print("Working on: " + line.strip('\n\r'))
	dataPath = dataDir + line.strip('\n\r')
	eventPath = eventDir + line.strip('\n\r')
	split_csv(dataPath, eventPath)	

'''
            print(
                "\n row_num: "
                + str(row_num)
                + " T: "
                + currentRowT
                + " start Row: "
                + str(start_row)
                + ", end row: "
                + str(end_row)
                + "\n"
            )
'''
print("Done")
