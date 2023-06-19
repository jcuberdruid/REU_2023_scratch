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
	#sys.stdout = open('/dev/null', 'w') 
	#sys.stderr = open('/dev/null', 'w') 
	for subject in range(subjectRange[0], subjectRange[1]+1):
		print("starting subject "+str(subject))
		for run in range(3,15):
			##load dataset
			raw = data.loadEEG(subject, run)
			raw = data.preproccess(subject, run, raw)	
			saveName = 'S'+str(subject)+'_'+str(run)+'.edf'
			thisSavePath = os.path.join(savePath, saveName)
			mne.export.export_raw(thisSavePath, raw, fmt='auto', physical_range='auto', add_ch_type=False, overwrite=True)#, verbose=None)
			data.epoches(subject, run, raw)
			print("done with epoches")
			print("exported subject"+str(subject)+", run: "+str(run))	
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
	#raw = data.loadEEG(1, 3)

	preprocessSave((1, 109))
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
