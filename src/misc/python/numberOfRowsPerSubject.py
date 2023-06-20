import csv

def count_rows_per_subject(csv_file):
    rows_per_subject = {}

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            subject = row['subject']
            if subject in rows_per_subject:
                rows_per_subject[subject] += 1
            else:
                rows_per_subject[subject] = 1

    return rows_per_subject

csv_file = '../data/datasets/sequences/MI_FF_T1_annotation.csv'
rows_per_subject = count_rows_per_subject(csv_file)

# Sort subjects in ascending order
sorted_subjects = sorted(rows_per_subject.keys())

for subject in sorted_subjects:
    count = rows_per_subject[subject]
    print(f"Subject {subject} has {count} rows.")
print(len(sorted_subjects))
