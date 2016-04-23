from collections import defaultdict 
import sys
import numpy as np

# Read Labels into an array
def readClass(fileName):
    with open(fileName, 'r') as f:
        labels = list()
        for line in f:
            labels.append(line.strip())
    return labels

def build_data_cv(fileName, cv = 10, classSelect = True):
    """
    Loads data and split into 10 folds.
    Default classSelect is True: 6 catagories.
	If `classSelect` set False: 50 catagories.
    """
    fineLabelArray = readClass('raw_data/fine.txt')
    classLabelArray = readClass('raw_data/class.txt')
    revs = []
    vocab = defaultdict(float)
    
    with open(fileName, "rb") as f:
        for sentence in f:
            sentence = sentence.decode("utf-8")
            labelClass = sentence.strip().split()[0].split(':')[0]
            labelFine = sentence.strip().split()[0]
            words = sentence.strip().split()[1:]
            wordsSet = set(words)
            for word in wordsSet:
                vocab[word] += 1
            if classSelect:
                label = classLabelArray.index(labelClass)
            else:
                label = fineLabelArray.index(labelFine)
            datum = {"y": label,
                     "text": " ".join(words),
                     "num_words": len(words),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    return revs, vocab

def getW(fileName, wordNum):
    w = np.zeros((wordNum, 301),dtype=np.float32)
    with open(fileName) as f:
        for i,line in enumerate(f):
            w[i,:]=line.split()
    return w

def getWordIdxMap(fileName):
    word_idx_map = dict()
    index = 0
    with open(fileName, 'r') as f:
        for line in f:
            word_idx_map[line[ : len(line) - 1]] = index
            index += 1
    return word_idx_map

def main():
    args = sys.argv[1]
    revs, vocab = build_data_cv(args)
    suffix = args.split('/')[-1]
    w = getW('w/' + suffix, len(vocab))
    word_idx_map = getWordIdxMap('index/' + suffix)
    print('word_idx_map: ' + str(len(word_idx_map)))
    print('w: ' + str(len(w)) + ' col: ' + str(len(w[0])))
    print('vocab: ' + str(len(vocab)))
    print('revs: ' + str(len(revs)))

if __name__ == "__main__":
    main()
