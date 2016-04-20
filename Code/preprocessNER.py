import nltk,os
import nltk.tag.stanford as st
import gensim, logging

# Set Environment
os.environ["CLASSPATH"] = "/Users/Lucifer/Documents/GraduateStudy/NLP/Trust-Filters/Code/stanford-ner-2014-06-16/stanford-ner.jar"
os.environ["STANFORD_MODELS"] = "/Users/Lucifer/Documents/GraduateStudy/NLP/Trust-Filters/Code/stanford-ner-2014-06-16/classifiers/"

#st1 = st.StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
#st2 = st.StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz')
st3 = st.StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz')

#Labelled data
fileName = "train.txt"

#retrieve named entity recognizer information from tagged sentences, and combine them.
def retrieve(strings):
    res = ""
    for string in strings:
        if string[1] != 'O':
            res += string[0] + "_"
    return res[:len(res) - 1]

with open(fileName, 'r') as fileInput:
    sentences = []
    for sentence in fileInput:
        words = sentence.split()
        strings = st3.tag(words)
        sentence = [words[0]] + [retrieve(strings)] + words[1:]
        sentences.append(sentence)
fileInput.close()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.save('myModel')
# new_model = gensim.models.Word2Vec.load('myModel')

# Output our results into file
fileOutput = open('test.txt', 'w')
for sentence in sentences:
    for word in sentence:
        stringArray = []
        for num in model[word]:
            stringArray.append('{:2s}'.format(str(num)))
        fileOutput.write(" ".join(stringArray))
        fileOutput.write('\n')
    fileOutput.write("\n\n")
fileOutput.close()

model.init_sims(replace=True)
