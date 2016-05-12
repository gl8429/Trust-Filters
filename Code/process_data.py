# -*- coding: utf-8 -*-

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
            words = clean_sentence(words)
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

def getW(fileName, wordNum, k=301):
    w = np.zeros((wordNum, k),dtype=np.float32)
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

# This function rescales word vectors last dimension (the NER label dimension) from 0-7 to 0 - 7 * 0.125
def rescaleNerLabel(w, k=301):
    size = len(w)
    for i in range(size):
        w[i][k-1] *= 0.125
    return w

# This function rescales word vectors to [-1, 1]
def rescaleWordVectors(w, k=301):
    size = len(w)
    for i in range(size):
        min_value = min(w[i][0:k-2])
        max_value = max(w[i][0:k-2])
        if max_value - min_value != 0:
            w[i][0:k-2] = (w[i][0:k-2] - min_value) / (max_value - min_value) * 2.0 - 1.0
        w[i][k-1] *= 0.125 #TODO: It's hard to determine what should we do to NER labels. Just rescale here
    return w

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
        elif word in w_idx_map2:
            W[i] = w2[w_idx_map2[word]]
            word_idx_map[word] = i
            i += 1
            continue
    #print i, len(W) i should equal len(W) + 1
    return W, word_idx_map

# For debug
def checkConsistencyWordIdxMapVocab(idx_map, vocab):
    ret = True
    for word in vocab:
        if word not in idx_map:
            ret = False
            print(word) #should not print anything if program is right
    return ret

# clean string in the sentences
def clean_sentence(sentence):
    result = list()
    for word in sentence:
        result.append(clean_str(word))
    return result

# clean `/` in the string and join them by `-` as processing in calculateVector.py before
def clean_str(string):
    if '/' in string:
        return '-'.join(string.split('/'))
    else:
         return string

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
    W = rescaleNerLabel(W) #uncomment this line if you want to rescale NER label from 0-7 to 0-7*125
    #W = rescaleWordVectors(W) #uncomment this line if you want to rescale whole vector

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    
    #Bug: testcase fail
    #print(checkConsistencyWordIdxMapVocab(train_word_idx_map, train_vocab))
    #print(checkConsistencyWordIdxMapVocab(test_word_idx_map, test_vocab))
    
    cPickle.dump([revs, W, W2, word_idx_map, vocab, selectClass], open("mr.p", "wb"))

if __name__ == "__main__":
    main()
