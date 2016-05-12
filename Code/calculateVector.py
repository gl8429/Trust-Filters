# -*- coding: utf-8 -*-

import sys
import nltk,os
import nltk.tag.stanford as st
import gensim, logging
import re

r""" This code is to calculate word's vectors and prepare arguments
     for preprocess.py. We add 1 more bit of previous 300 bits vector.
     Additionally, we also prepare a vocuburary of all words, word
     index map and all words vectors
"""


# Enlarge the memory set of JAVA
nltk.internals.config_java(options='-xmx2G')

# Set Environment Config of Standford NER Parser
os.environ["CLASSPATH"] = "/Users/Lucifer/Documents/GraduateStudy/NLP/stanford-package/stanford-ner-2014-06-16/stanford-ner.jar"
os.environ["STANFORD_MODELS"] = "/Users/Lucifer/Documents/GraduateStudy/NLP/stanford-package/stanford-ner-2014-06-16/classifiers/"

# Train Stanford NER tagger, which has three options

#st1 = st.StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
#NER = ['O', 'LOCATION', 'PERSON', 'ORGANIZATION']
#st2 = st.StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz')
#NER = ['O', 'LOCATION', 'PERSON', 'ORGANIZATION', 'MISC']
st = st.StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz')

# The NER class list
NER = ['O', 'TIME', 'LOCATION', 'ORGANIZATION', 'PERSON', 'MONEY', 'PERCENT', 'DATE']

def main():
    args  = sys.argv
    inFile = args[1]
    prefix = 'vectors/'
    suffix = args[1].split('/')[-1]
    outFile = prefix + suffix
    #nerFile = 'ner/' + suffix
    indexFile = 'index/' + suffix
    wFile = 'w/' + suffix

    # Store all NER in the sentence with that name entity
    NERDict = dict()

    with open(inFile, 'r') as fileInput:
        sentences = list()
        for sentence in fileInput:
            originSentence = clean_sentence(sentence.split()[1:])

            # Stanford NER tagger
            strings = st.tag(originSentence)
            for string in strings:
                NERDict[string[0]] = NER.index(string[1])
            sentences.append(originSentence)
    fileInput.close()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Call Word2Vector to train words and get the model
    model = gensim.models.Word2Vec(sentences, size=300, window=5, min_count=1, workers=4)
    model.save('model/' + suffix.split('.')[0])


    index = 0
    # Prepare for word index map
    word_idx_map = dict()
    w = list()

    # Output all the results into files
    fileOutput = open(outFile, 'w')
    wOutput = open(wFile, 'w')
    iFile = open(indexFile, 'w')

    for sentence in sentences:
        for word in sentence:
            stringArray = []

            # Append all vectors of a particular word
            for num in model[word]:
                stringArray.append('{:2s}'.format(str(num)))

            # Then concatanate with 1-bit additional NER vector
            stringArray.append('{:2s}'.format(str(NERDict[word])))
            outputStream = " ".join(stringArray) + '\n'

            # Store word into word index map
            if word not in word_idx_map:
                w.append(outputStream)
                wOutput.write(outputStream)
                iFile.write(word + '\n')
                word_idx_map[word] = index
                index += 1
            fileOutput.write(outputStream)
        fileOutput.write("\n")
    fileOutput.close()
    wOutput.close()
    iFile.close()
    model.init_sims(replace=True)

    #fileOutput = open(nerFile, 'w')

r""" Clean some unused character in the string

Parameters
----------
string : str
         A sentence of the original string

Returns
-------
string : str
         A string replaced '/' by '-'

"""

def clean_str(string):
    if '/' in string:
        return '-'.join(string.split('/'))
    else:
        return string

r""" Clean all string in a sentence
Parameters
----------
sentence : list
          A list of string

Returns
-------
result : list
         A processed string

"""

def clean_sentence(sentence):
    result = list()
    for word in sentence:
        result.append(clean_str(word))
    return result

if __name__ == "__main__":
    main()
