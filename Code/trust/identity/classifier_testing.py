import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import nltk
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize import (RegexpTokenizer, TreebankWordTokenizer,
                           WordPunctTokenizer, WhitespaceTokenizer,
                           word_tokenize)

if __name__ == '__main__':
    path_to_jar = '/vagrant/stanford_nlp/stanford-corenlp-3.6.0.jar'
    stanford_tokenizer = StanfordTokenizer(path_to_jar=path_to_jar)
    regex_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    treebank_tokenizer = TreebankWordTokenizer()
    wordpunct_tokenizer = WordPunctTokenizer()
    whitespace_tokenizer = WhitespaceTokenizer()

    pwd = os.path.abspath(os.path.dirname(__file__))

    # Step 1
    #   Read the labeled data into memory.
    me_df = pd.DataFrame()
    with open(os.path.join(pwd, 'labeled_data/ME.txt'), 'rb') as f:
        for line in f:
            raw = line
            stanford = stanford_tokenizer.tokenize(raw)
            regex = regex_tokenizer.tokenize(raw)
            treebank = treebank_tokenizer.tokenize(raw)
            wordpunct = wordpunct_tokenizer.tokenize(raw)
            whitespace = whitespace_tokenizer.tokenize(raw)
            cv = CountVectorizer()
            cv.fit_transform([line])
            sklearn = cv.get_feature_names()
            me_df = me_df.append(\
                {
                    'raw': raw,
                    'stanford': stanford,
                    'regex': regex,
                    'treebank': treebank,
                    'wordpunct': wordpunct,
                    'whitespace': whitespace,
                    'sklearn': sklearn,
                    'label': 'me'
                },
                ignore_index=True)


    notme_df = pd.DataFrame()
    with open(os.path.join(pwd, 'labeled_data/NOT_ME.txt'), 'rb') as f:
        for line in f:
            raw = line
            stanford = stanford_tokenizer.tokenize(raw)
            regex = regex_tokenizer.tokenize(raw)
            treebank = treebank_tokenizer.tokenize(raw)
            wordpunct = wordpunct_tokenizer.tokenize(raw)
            whitespace = whitespace_tokenizer.tokenize(raw)
            cv = CountVectorizer()
            cv.fit_transform([line])
            sklearn = cv.get_feature_names()
            notme_df = notme_df.append(\
                {
                    'raw': raw,
                    'stanford': stanford,
                    'regex': regex,
                    'treebank': treebank,
                    'wordpunct': wordpunct,
                    'whitespace': whitespace,
                    'sklearn': sklearn,
                    'label': 'notme'
                },
                ignore_index=True)



    # Step 2
    #   Partition the labeled data into a training set and a testing set.
    data_split_idx = lambda percent,df: int(np.floor((float(percent)/100)*len(df)))
    me_idx = data_split_idx(75, me_df)
    notme_idx = data_split_idx(75, notme_df)
    training_df = pd.concat([me_df[:me_idx], notme_df[:notme_idx]],
                            ignore_index=True)
    testing_df = pd.concat([me_df[me_idx:], notme_df[notme_idx:]],
                           ignore_index=True)
    training_df = training_df.reindex(np.random.permutation(training_df.index))
    testing_df = testing_df.reindex(np.random.permutation(testing_df.index))







    # Step 3
    #   Preprocess the data.
    def preprocess(**kwargs):
        token_pattern = u'(?u)\\b\\w+\\b'
        if kwargs['unigram']:
            if kwargs['kind'] == 'count':
                return CountVectorizer(token_pattern=token_pattern)
            elif kwargs['kind'] == 'freq':
                return TfidfVectorizer(token_pattern=token_pattern, use_idf=False)
            elif kwargs['kind'] == 'tfidf':
                return TfidfVectorizer(token_pattern=token_pattern)
        if kwargs['bigram']:
            if kwargs['kind'] == 'count':
                return CountVectorizer(token_pattern=token_pattern,
                                       ngram_range=(1,2))
            elif kwargs['kind'] == 'freq':
                return TfidfVectorizer(token_pattern=token_pattern, use_idf=False,
                                       ngram_range=(1,2))
            elif kwargs['kind'] == 'tfidf':
                return TfidfVectorizer(token_pattern=token_pattern,
                                       ngram_range=(1, 2))

    # Unigram counts.
    unigram_count_vect = preprocess(unigram=True, kind='count')
    unigram_count_trans = unigram_count_vect.fit_transform(training_df.text)
    unigram_count_vocab = unigram_count_vect.vocabulary_
    unigram_count_dict = {\
        w: pd.Series(unigram_count_trans.getcol(unigram_count_vocab[w]).data)
        for w in unigram_count_vocab}
    unigram_count_df = pd.DataFrame(unigram_count_dict).fillna(0)

    # Bigram counts.
    bigram_count_vect = preprocess(bigram=True, kind='count')
    bigram_count_trans = bigram_count_vect.fit_transform(training_df.text)
    bigram_count_vocab = bigram_count_vect.vocabulary_
    bigram_count_dict = {\
        w: pd.Series(bigram_count_trans.getcol(bigram_count_vocab[w]).data)
        for w in bigram_count_vocab}
    bigram_count_df = pd.DataFrame(bigram_count_dict).fillna(0)

    # Unigram frequencies.
    unigram_freq_vect = preprocess(unigram=True, kind='freq')
    unigram_freq_trans = unigram_freq_vect.fit_transform(training_df.text)
    unigram_freq_vocab = unigram_freq_vect.vocabulary_
    unigram_freq_dict = {\
        w: pd.Series(unigram_freq_trans.getcol(unigram_freq_vocab[w]).data)
        for w in unigram_freq_vocab}
    unigram_freq_df = pd.DataFrame(unigram_freq_dict).fillna(0)

    # Bigram frequencies.
    bigram_freq_vect = preprocess(bigram=True, kind='freq')
    bigram_freq_trans = bigram_freq_vect.fit_transform(training_df.text)
    bigram_freq_vocab = bigram_freq_vect.vocabulary_
    bigram_freq_dict = {\
        w: pd.Series(bigram_freq_trans.getcol(bigram_freq_vocab[w]).data)
        for w in bigram_freq_vocab}
    bigram_freq_df = pd.DataFrame(bigram_freq_dict).fillna(0)

    # Unigram tfidf.
    unigram_tfidf_vect = preprocess(unigram=True, kind='tfidf')
    unigram_tfidf_trans = unigram_tfidf_vect.fit_transform(training_df.text)
    unigram_tfidf_vocab = unigram_tfidf_vect.vocabulary_
    unigram_tfidf_dict = {\
        w: pd.Series(unigram_tfidf_trans.getcol(unigram_tfidf_vocab[w]).data)
        for w in unigram_tfidf_vocab}
    unigram_tfidf_df = pd.DataFrame(unigram_tfidf_dict).fillna(0)

    # Bigram tfidf.
    bigram_tfidf_vect = preprocess(bigram=True, kind='tfidf')
    bigram_tfidf_trans = bigram_tfidf_vect.fit_transform(training_df.text)
    bigram_tfidf_vocab = bigram_tfidf_vect.vocabulary_
    bigram_tfidf_dict = {\
        w: pd.Series(bigram_tfidf_trans.getcol(bigram_tfidf_vocab[w]).data)
        for w in bigram_tfidf_vocab}
    bigram_tfidf_df = pd.DataFrame(bigram_tfidf_dict).fillna(0)














    # Step 4
    #   Create a classifier.
    classifier = MultinomialNB().fit(training_tfidf, training_df.label)

    # Step 5
    #   Test how well the classifier does on the testing data.
    pipe


    token_pattern = u'(?u)\\b\\w+\\b'
    unigram_count = CountVectorizer(token_pattern=token_pattern)
    unigram_freq = TfidfVectorizer(token_pattern=token_pattern, use_idf=False)
    unigram_tfidf = TfidfVectorizer(token_pattern=token_pattern)
    bigram_count = CountVectorizer(token_pattern=token_pattern, ngram_range=(1,2))
    bigram_freq = TfidfVectorizer(token_pattern=token_pattern, use_idf=False,
                                  ngram_range=(1,2))
    bigram_tfidf = TfidfVectorizer(token_pattern=token_pattern, ngram_range=(1, 2))



    pipe = lambda x: Pipeline([('vect', x), ('clf', MultinomialNB())])
    clf = pipe(unigram_count).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(unigram_freq).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(unigram_tfidf).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(bigram_count).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(bigram_freq).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(bigram_tfidf).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    # How is bigram counting the best?
    pipe = lambda x: Pipeline([('vect', x), ('clf', SGDClassifier(loss='hinge',
        penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
    clf = pipe(unigram_count).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(unigram_freq).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(unigram_tfidf).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(bigram_count).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(bigram_freq).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    clf = pipe(bigram_tfidf).fit(training_df.text, training_df.label)
    prediction = clf.predict(testing_df.text)
    print np.mean(prediction == testing_df.label)

    # Grid search
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
    }
    clf = SGDClassifier()
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)





    path = '/vagrant/local_datastore'
    all_files = [os.path.abspath(os.path.join(p, F))
                 for p,d,f in os.walk(path)
                 for F in f if F.endswith('.json')]



















    # import os
    # import shutil
    # import sys

    # import numpy as np
    # from bs4 import BeautifulSoup as bs


    # file_paths_file = '/vagrant/bsve_datastore/data/blogs/wordpress/file_paths.txt'
    # local_datastore_path = '/vagrant/local_datastore'

    # random_full_history = '/vagrant/bsve_datastore/data/blogs/wordpress/deadcitizensrightssociety.wordpress.com'

    # file_paths = []


    # with open(file_paths_file, 'rb') as f:
    #     for line in f:
    #         file_paths.append(line.replace('/Volumes',
    #                                        '/vagrant').replace('\n', ''))

    # random_file_sample = np.random.permutation(file_paths)[:1000]


    # nbr_posts = []
    # for i,rf in enumerate(random_file_sample):
    #     directory = os.path.dirname(rf)
    #     nbr_posts.append((len(os.listdir(directory)), directory))
    #     percent = float(i + 1)/1000*100
    #     sys.stdout.flush()
    #     sys.stdout.write('\r%0.2f%% completed.' % percent)


    # for rf in random_file_sample:
    #     filename = os.path.basename(rf)
    #     directory = '/'.join(os.path.dirname(rf).split('/')[3:])
    #     save_directory = os.path.join(local_datastore_path, directory)
    #     save_path = os.path.join(save_directory, filename)
    #     if not os.path.exists(save_directory):
    #         os.makedirs(save_directory)
    #     try:
    #         shutil.copy(rf, save_path)
    #     except:
    #         print 'shit!'




















    import nltk
    from nltk.corpus import movie_reviews
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    # Create a list of file paths to labeled movie reviews from NLTK.
    pos_reviews = movie_reviews.fileids('pos')
    neg_reviews = movie_reviews.fileids('neg')

    # Create pandas data frames for the training and test datasets.
    pos_train_df = pd.DataFrame()
    neg_train_df = pd.DataFrame()
    pos_test_df = pd.DataFrame()
    neg_test_df = pd.DataFrame()
    for i,f in enumerate(zip(pos_reviews, neg_reviews)):
        if i < 750:
            pos_train_df = pos_train_df.append(
                    {'text': movie_reviews.raw(fileids=f[0]),
                     'label': 'pos'}, ignore_index=True)
            neg_train_df = neg_train_df.append(
                    {'text': movie_reviews.raw(fileids=f[1]),
                     'label': 'neg'}, ignore_index=True)
        else:
            pos_test_df = pos_test_df.append(
                    {'text': movie_reviews.raw(fileids=f[0]),
                     'label': 'pos'}, ignore_index=True)
            neg_test_df = neg_test_df.append(
                    {'text': movie_reviews.raw(fileids=f[1]),
                     'label': 'neg'}, ignore_index=True)
    training_df = pd.concat([pos_train_df, neg_train_df], ignore_index=True)
    testing_df = pd.concat([pos_test_df, neg_test_df], ignore_index=True)
    training_df = training_df.reindex(np.random.permutation(training_df.index))
    testing_df = testing_df.reindex(np.random.permutation(testing_df.index))

    # Training tfidf.
    # Need to adjust the token_pattern to be u'(?u)\\b\\w+\\b' in order to retain
    # single letter words.
    tfidf_vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    training_tfidf = tfidf_vectorizer.fit_transform(training_df.text)
    training_vocabulary = tfidf_vectorizer.vocabulary_
    training_tfidf_dict = {w: pd.Series(training_tfidf.getcol(vocabulary[w]).data)
                           for w in vocabulary}
    training_tfidf_df = pd.DataFrame(training_tfidf_dict).fillna(0)

    # Testing tfidf.
    tfidf_vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    testing_tfidf = tfidf_vectorizer.fit_transform(testing_df.text)
    testing_vocabulary = tfidf_vectorizer.vocabulary_
    testing_tfidf_dict = {w: pd.Series(testing_tfidf.getcol(vocabulary[w]).data)
                          for w in vocabulary}
    testing_tfidf_df = pd.DataFrame(testing_tfidf_dict).fillna(0)

    # Classifier.
    classifier = MultinomialNB().fit(training_tfidf, training_df.text)



