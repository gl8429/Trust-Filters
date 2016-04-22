fileName = "raw_data/train5500.txt"
with open(fileName, 'r') as fileInput:
    catagoriesSet = set()
    finesSet = set()
    for sentence in fileInput:
        finesSet.add(sentence.split(' ')[0])
        catagoriesSet.add(sentence.split(' ')[0].split(':')[0])
    catagories = list()
    fines = list()
    for fine in finesSet:
        fines.append(fine)
    fines.sort()
    for cato in catagoriesSet:
        catagories.append(cato)
    catagories.sort()

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
