order of operations:

1. make trials: download and export csvs for each of the 12 trials per subject
2. concatTasks: merge csvs based on subject and task. Ex csvs 3, 7, 11 all task 1 
3. concatClasses: merge the concatTask csvs across subjects based on tasks 
4. create numpy sequences from the class csvs 
5. profit 
