# CLI 

# -test: which testingSubject(s) to use 
# -train: which training subjects to use (if not using clustering)
# -lsdata: lists available datasets enumerated 
# -data: used to select which enumerated dataset to use 
# -lscluster: list available clustersets enumerated 
# -cluster: used to select which enumerated clusterset to use 
# -lognote or ln: string to include in classification output logs

import argparse
import sys
import subprocess
import json
from utils import config

def parse_arguments():
    parser = argparse.ArgumentParser(description='Program to process and classify data.')

    parser.add_argument('-setConfig', dest='set_config', action='store_true',
                    help='calls the set config utility')
    parser.add_argument('-test', 
                        dest='test_subjects',
                        nargs='+',
                        help='Specify the testing subjects to use.')

    parser.add_argument('-lognote', '-ln',
                        dest='log_note',
                        type=str,
                        help='Include a string note in the classification output logs.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    # Rest of the code to process the args and run accordingly
    # For example, if you want to print the test subjects:
    if args.test_subjects:
        print(f'Testing subjects: {args.test_subjects}')
    if args.set_config:
        config.setConfig()
        quit()
    #get defaults 
    defaults = config.readConfig()
    #call subEpochs with testSubject, model, dataset 
    json_test_subjects = json.dumps(args.test_subjects)
    #subprocess.run(['python3', './classification/tuned_cluster_HP_Search_classify.py', json_test_subjects, defaults['model'], defaults['dataset'], defaults['clusterset']])
    subprocess.run(['python3', './classification/2dfft_classify.py', json_test_subjects, defaults['model'], defaults['dataset'], defaults['clusterset']])
# -v: verbose but in what way? (which subject working on vs epoch performance (could make custom callbacks to do both + env variables)
# -q: send all output to /dev/null
# -help
