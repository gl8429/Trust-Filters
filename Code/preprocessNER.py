import nltk,os
import nltk.tag.stanford as st
import gensim, logging

# Set Environment
os.environ["CLASSPATH"] = "/Users/Lucifer/Documents/GraduateStudy/NLP/Trust-Filters/Code/stanford-ner-2014-06-16/stanford-ner.jar"
os.environ["STANFORD_MODELS"] = "/Users/Lucifer/Documents/GraduateStudy/NLP/Trust-Filters/Code/stanford-ner-2014-06-16/classifiers/"

#st1 = st.StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
#st2 = st.StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz')
st = st.StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz')

fileName = "train.txt"
with open(fileName, 'r') as f:
    sentences = []
    for line in f:
        strings = st3.tag(line.split())
        line = [retrieve(strings)] + line.split()
        sentences.append(line)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.save('myModel')
# new_model = gensim.models.Word2Vec.load('myModel')

model.init_sims(replace=True)


#retrieve named entity recognizer information from tagged sentences, and combine them.
def retrieve(strings):
    res = ""
    for string in strings:
        if string[1] != 'O':
            res += string[0] + "_"
    return res[:len(res) - 1]
