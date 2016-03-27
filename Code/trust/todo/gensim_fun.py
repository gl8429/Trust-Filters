import os
import json

from nltk.corpus import stopwords
import gensim
from bioevent.trust import utils
import langdetect

# Load some sample data into memory.
basepath = '/vagrant/bsve_datastore/blogs/wordpress/done/'
blogdirs = [os.path.join(basepath, path) for path in os.listdir(basepath)]
fetch = utils.FetchTestData(basepath)
data = fetch.raw()

# Flu data
flu = [os.path.join(path, 'flu_posts_tokens.json') for path in blogdirs
       if os.path.exists(os.path.join(path, 'flu_posts_tokens.json'))]
length = len(flu)
flu_docs = []
for i,f in enumerate(flu):
    with open(f, 'r') as F:
        data = json.load(F)
    for k,v in data.items():
        flu_docs.append(v)
    print '{0}/{1}'.format(i+1, length)
data = [filter(lambda x: x not in stop_words and x.isalpha(), tokens)
        for tokens in flu_docs if tokens]
data = [tokens for tokens in data if tokens]
data = [tokens for tokens in data
        if langdetect.detect(' '.join(tokens)) == 'en']
dictionary = gensim.corpora.Dictionary(data)
corpus = [dictionary.doc2bow(text) for text in data]
# HDP
hdp = gensim.models.HdpModel(corpus, id2word=dictionary)
hdp.print_topics(topics=10, topn=10)




# Gensim fun.
stop_words = stopwords.words('english')
data = {
    k: [token for token in v if token.isalpha() and token not in stop_words]
    for k,v in raw_data.items()
}
documents = [data[k] for k in data]
dictionary = gensim.corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]


# HDP
hdp = gensim.models.HdpModel(corpus, id2word=dictionary)
hdp.print_topics(topics=10, topn=10)

# LDA
glda = gensim.models.LdaModel(corpus,
                              id2word=dictionary,
                              num_topics=3,
                              passes=2,
                              iterations=100,
                              update_every=5,
                              eval_every=5,
                              chunksize=5)
glda.print_topics()
