import json
from dataclasses import dataclass

@dataclass
class TrainingClass:
    fileName: str
    testingIndices: []
    trainingIndices: []
    similarSubjectsEpochs: dict = None
    similarIndices: [] = None

    def __init__(self, fileName: str, testingIndices: [], trainingIndices: [], similarIndices: [] = None):
        self.fileName = fileName
        self.testingIndices = testingIndices
        self.trainingIndices = trainingIndices
        self.similarIndices = similarIndices

    def appendSimilar(self, subject: int, epochs: [], indices: []):
        if self.similarSubjectsEpochs is None:
            self.similarSubjectsEpochs = {}
        self.similarSubjectsEpochs[str(subject)] = epochs
        if self.similarIndices is None:
            self.similarIndices = []
        self.similarIndices.extend(indices)

class Subject:
    subject: int
    classes: []

    def __init__(self, subject: int):
        self.subject = subject
        self.classes = []

    def appendClasses(self, trainingClass: TrainingClass):
        self.classes.append(trainingClass)

    def toJson(self):
        class_data = {}
        class_data["subject"] = self.subject
        for i, training_class in enumerate(self.classes):
            class_data[f"class_{i+1}"] = {
                "filename": training_class.fileName,
                "testingIndices": training_class.testingIndices,
                "trainingIndices": training_class.trainingIndices,
                "similarSubjects": training_class.similarSubjectsEpochs,
                "similarIndices": training_class.similarIndices,
            }
        return json.dumps(class_data, indent=4)


'''

{"subject": 1,
"class_1": {
"filename" : "MI_RLH_T1.npy",
"testingIndices": [x, y, z, 1, 2, 3],
"trainingIndices": [x, y, z, 1, 2, 3],
"similarSubjects": ["1":[2,4,5,9],"2":[8,7,9,2], "3":[12, 1, 3, 5]]
},
"class_2": {
"filename" : "MM_RLH_T1.npy",
"testingIndices": [x, y, z, 1, 2, 3],
"trainingIndices": [x, y, z, 1, 2, 3],
"similarSubjectsEpochs": ["1":[2,4,5,9],"2":[8,7,9,2], "3":[12, 1, 3, 5]]
},
}
subject = Subject(1)
class_1 = TrainingClass("MI_RLH_T1.npy", [1, 2, 3], [1, 2, 3])
class_1.appendSimilar(1, [2, 4, 5, 9], [1,2,3,4,5,6])
class_1.appendSimilar(2, [2, 4, 5, 9], [1,2,3,4,5,6])
class_1.appendSimilar(3, [2, 4, 5, 9], [1,2,3,4,5,6])
subject.appendClasses(class_1)

class_2 = TrainingClass("MM_RLH_T1.npy", [1, 2, 3], [1, 2, 3])
class_2.appendSimilar(2, [2, 4, 5, 9], [1,4,5,6,2,7])
class_2.appendSimilar(3, [2, 4, 5, 9], [1,4,5,6,2,7])
subject.appendClasses(class_2)

class_3 = TrainingClass("MM_RLH_T1.npy", [1, 2, 3], [1, 2, 3])
class_3.appendSimilar(8, [2, 4, 5, 9], [1,4,5,6,2,7])
class_3.appendSimilar(3, [2, 4, 5, 9], [1,4,5,6,7])
subject.appendClasses(class_3)

json_string = subject.toJson()
print(json_string)
'''
