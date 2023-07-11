import os 
from simple_term_menu import TerminalMenu

#relative paths 
utilsPath = os.path.dirname(os.path.realpath(__file__))
models = utilsPath+"/../classification/models/"
data = utilsPath+"/../../data/datasets/"
logs = utilsPath+"/../../logs/"
clustering_logs = logs + "clustering_logs/"
tf_logs = logs + "tf_perf_logs/"

#return list directories sorted by creation time
def list_directories(path):
    directories = [(d, os.path.getctime(os.path.join(path, d))) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    sorted_directories = sorted(directories, key=lambda x: x[1])
    return [d[0] for d in sorted_directories]

def list_files(path):
    files = os.listdir(path) 
    sorted_files = sorted(files, key=lambda x: x[1])
    for x in sorted_files:
        x = os.path.join(path, x)
    return sorted_files

