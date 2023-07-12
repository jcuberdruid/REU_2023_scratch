'''

The problem: currently my clustering is set to use subepochs, but that causes an issue because the 50% overlap could
cause training data corruption, and might complicate clustering 

Solution: the goal of this script is to take in subepochs and create numpy arrays of epochs denoted by 'chunks' 
in the annotations file, so that it can be used with my clustering script

possible issues:
- clustering logs, and the classifier that reads them are set to use "similar indices" (aka subepochs), although they 
also save similar subjects and epochs so either we convert back to the subepochs (for each index multiply it by 15 
and add the next 14 numberse to convert to subepochs, or we leave the "similar indices" blank and load things based upon epoch in the classifier. The former option seems better to conform and limit changes to one file. 

input: data.npy(n, 80, 17, 17), annotations.csv
output: dataEpochs.npy(n/15, 1200, 17, 17), annotationsEpochs.npy
'''
#load data
#load annotation



