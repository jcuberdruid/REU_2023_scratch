import os
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
## MNE proccessing
import mne
import data
from multiprocessing import Process, freeze_support

savePath = "./proccessedDB"

def preprocessSave(subjectRange):
#	sys.stdout = open('/dev/null', 'w') 
#	sys.stderr = open('/dev/null', 'w') 
	for subject in range(subjectRange[0], subjectRange[1]+1):
		print("starting subject "+str(subject))
		for run in range(4,15):
			##load dataset
			raw = data.loadEEG(subject, run)
			print("exported subject Events "+str(subject)+", run: "+str(run))	
	return

def divide_tasks(num_tasks, dividing_number):
    quotient, remainder = divmod(num_tasks, dividing_number)
    start = 1
    divided_tasks = []

    for i in range(dividing_number):
        end = start + quotient - 1
        if i < remainder:
            end += 1
        divided_tasks.append((start, end))
        start = end + 1

    return divided_tasks

def main():
	if(os.path.exists(savePath) != True):
		os.mkdir(savePath)
	concurrencyMult = 5
	tasks_divided = divide_tasks(109, concurrencyMult)

	print("concurrency set to process " + str(concurrencyMult) + " subjects at a time")
	print("preprocessing PhysioNet database: 109 subjects, 14 records per subject")

	preprocessSave((1, 87))
	preprocessSave((89, 91))
	preprocessSave((93, 99))
	preprocessSave((101, 103))
	preprocessSave((105, 109))
	'''
	processes = []
	for x in range(concurrencyMult): 
		p = Process(target=preprocessSave, args=(tasks_divided[x],))
		p.start()
		processes.append(p)
	# Wait for all processes to finish
	for p in processes:
		p.join()
	'''
if __name__ == "__main__":
    freeze_support()
    main()
