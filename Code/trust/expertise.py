# -*- coding: utf-8 -*-

r"""
"""

from bioevent.trust import utils

# Load some stop words from NLTK.
# NOTE this will need to be adjusted for words like 'I', 'me', etc.
stop_words = stopwords.words('english')

# Load some sample data into memory.
basepath = '/vagrant/bsve_datastore/blogs/wordpress/done/'
blogdirs = [os.path.join(basepath, path) for path in os.listdir(basepath)]
fetch = utils.FetchTestData(basepath)
data = fetch.raw()






# Flu data
flupath = lambda x: os.path.join(x, 'flu_posts_tokens.json')
flu = [flupath(path) for path in blogdirs if os.path.exists(flupath(path))]
length = len(flu)
flu_docs = []
for i,f in enumerate(flu):
    with open(f, 'r') as F:
        data = json.load(F)
    for k,v in data.items():
        tokens = filter(lambda x: x not in stop_words and x.isalpha(),
                        [t for t in v if t])
        if tokens:
            tokens = [t for t in tokens
                      if langdetect.detect(' '.join(t)) == 'en']
            flu_docs.append(tokens)
    print '{0}/{1}'.format(i+1, length)
data = [filter(lambda x: x not in stop_words and x.isalpha(), tokens)
        for tokens in flu_docs if tokens]
data = [tokens for tokens in data if tokens]
data = [tokens for tokens in data
        if langdetect.detect(' '.join(tokens)) == 'en']

# Create a gensim dictionary.
dictionary = gensim.corpora.Dictionary(data)
corpus = [dictionary.doc2bow(text) for text in data]

