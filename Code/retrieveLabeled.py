# -*- coding: utf-8 -*-

r""" Retrieve all classes and all fines and store into file

Output
------
raw_data/class.txt : all kinds of classes
raw_data/fine.txt : all kinds of fines

"""

# A full version of training data

fileName = "raw_data/train5500.txt"

with open(fileName, 'r') as fileInput:
    # Store it into a set
    catagoriesSet = set()
    finesSet = set()
    # Retrieve class name
    for sentence in fileInput:
        finesSet.add(sentence.split(' ')[0])
        catagoriesSet.add(sentence.split(' ')[0].split(':')[0])
    catagories = list()
    fines = list()
    for fine in finesSet:
        fines.append(fine)
    # Sort the catagory
    fines.sort()
    for cato in catagoriesSet:
        catagories.append(cato)
    catagories.sort()

# Store the data into file
classFile = 'raw_data/class.txt'
with open(classFile, 'w') as out:
    for catagory in catagories:
        out.write(catagory + '\n')
out.close()
fineFile = 'raw_data/fine.txt'
with open(fineFile, 'w') as out:
    for fine in fines:
        out.write(fine + '\n')
out.close()
