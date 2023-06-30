import pandas as pd

import os

def merge_csvs(files):
    df = pd.DataFrame()
    for index, file in enumerate(files):
        print()
        print(f"{index}: {file}")
        print()
        data = pd.read_csv(file)
        df = pd.concat([df, data], axis=0)
    return df

inTakeDir = "../../data/datasets/unproccessed/trials/csvs/"
outputDir = "../../data/datasets/unproccessed/tasks/"

for s in range(1, 110):
    if (s == 88 or s == 92 or s == 100 or s == 104):
        continue
    fileStart = f"S{s}_"
    print(fileStart)
    for e in range(1, 3):
        fileEnd = f"_T{e}.csv"
        task1Files = []
        task2Files = []
        task3Files = []
        task4Files = []
        for r in range(3,12,4):
            task1Files.append(inTakeDir+fileStart+str(r)+fileEnd)
        for r in range(4,13,4):
            task2Files.append(inTakeDir+fileStart+str(r)+fileEnd)
        for r in range(5,14,4):
            task3Files.append(inTakeDir+fileStart+str(r)+fileEnd)
        for r in range(6,15,4):
            task4Files.append(inTakeDir+fileStart+str(r)+fileEnd)
        #print(f"T{e}: ")
        #print(task1Files) 
        #print(task2Files) 
        #print(task3Files) 
        #print(task4Files) 
        task1Merged = merge_csvs(task1Files)
        task2Merged = merge_csvs(task2Files)
        task3Merged = merge_csvs(task3Files)
        task4Merged = merge_csvs(task4Files)
        task1Merged.to_csv(outputDir+fileStart+"MM_RLH"+fileEnd, index=False)
        task2Merged.to_csv(outputDir+fileStart+"MI_RLH"+fileEnd, index=False)
        task3Merged.to_csv(outputDir+fileStart+"MM_FF"+fileEnd, index=False)
        task4Merged.to_csv(outputDir+fileStart+"MI_FF"+fileEnd, index=False)
