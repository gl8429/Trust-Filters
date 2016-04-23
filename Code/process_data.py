from collections import defaultdict 
import sys
import numpy as np
import sys, re
import cPickle

# Read Labels into an array
def readClass(fileName):
    with open(fileName, 'r') as f:
        labels = list()
        for line in f:
            labels.append(line.strip())
    return labels

def build_data_cv(fileName, fileType, classSelect = True):
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
                     "split": fileType}
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

# This function add random vectors for those words occur in vocab at least min_df times but not in word_vecs
# Input:
#   word_vecs: {} dict: word_vecs[word_string] = a vector
#   vocab: dict vocab[word_string] = number of occursion of the word
#   min_df: the number of at least occur  
# Output: 
#   nothing
def add_unknown_words(word_vecs, vocab, min_df=1, k=301):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
 
# Input:
#   word_vecs: mapping from word_string to vector
# Output:
#   W: a matrix, W[i] is the vector for word indexed by i
#   word_idx_map: mapping from word_string to index i
 
def get_W(word_vecs, k=301):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def combineRevs(revs1, revs2):
    revs = []
    for rev in revs1:
	revs.append(rev)
    for rev in revs2:
        revs.append(rev)
    return revs

def combineVocab(vocab1, vocab2):
    vocab = defaultdict(float)
    for word in vocab1:
	if word in vocab2:          
            vocab[word] = vocab1[word] + vocab2[word]
        else:
            vocab[word] = vocab1[word]
    for word in vocab2:
	if word not in vocab1:
            vocab[word] = vocab2[word]
    return vocab

def combineWordVectors(w1, w2, w_idx_map1, w_idx_map2, vocab, k=301):
    vocab_size = len(vocab)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in vocab:
        if word in w_idx_map1:
	    W[i] = w1[w_idx_map1[word]]
            word_idx_map[word] = i
            i += 1
            continue
        if word in w_idx_map2:
            W[i] = w2[w_idx_map2[word]]
            word_idx_map[word] = i
            i += 1
            continue
    return W, word_idx_map

def main():
    trainFileName = sys.argv[1]
    testFileName = sys.argv[2]   
    selectClassStr = sys.argv[3]
    
    selectClass = True
    if selectClassStr == '-fine':
        selectClass = False

    train_revs, train_vocab = build_data_cv(trainFileName, 0, selectClass)
    test_revs, test_vocab = build_data_cv(testFileName, 1, selectClass)

    revs=combineRevs(train_revs, test_revs)
    vocab=combineVocab(train_vocab, test_vocab)

    train_suffix = trainFileName.split('/')[-1]
    train_W = getW('w/' + train_suffix, len(train_vocab))
    
    test_suffix = testFileName.split('/')[-1]
    test_W = getW('w/' + test_suffix, len(test_vocab))
 
    train_word_idx_map = getWordIdxMap('index/' + train_suffix)
    test_word_idx_map = getWordIdxMap('index/' + test_suffix)
    
    W, word_idx_map = combineWordVectors(train_W, test_W, train_word_idx_map, test_word_idx_map, vocab)

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab, selectClass], open("mr.p", "wb"))

if __name__ == "__main__":
    main()
