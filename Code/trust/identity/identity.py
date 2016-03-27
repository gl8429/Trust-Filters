import os
from ConfigParser import SafeConfigParser

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV

from bokeh.plotting import figure, show, ColumnDataSource, output_notebook
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, PanTool
from bokeh.models import ResizeTool, WheelZoomTool

import nltk
from nltk.tokenize import (RegexpTokenizer,
                           SpaceTokenizer,
                           TreebankWordTokenizer,
                           WhitespaceTokenizer,
                           WordPunctTokenizer,
                           stanford,
                           word_tokenize)

# Initialize the tokenizers.
path_to_jar = '/vagrant/stanford_nlp/stanford-corenlp-3.6.0-models.jar'
regex_tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
space_tokenizer = SpaceTokenizer()
treebank_tokenizer = TreebankWordTokenizer()
whitespace_tokenizer = WhitespaceTokenizer()
wordpunct_tokenizer = WordPunctTokenizer()
stanford_tokenizer = stanford.StanfordTokenizer(path_to_jar=path_to_jar)

def get_config():
    r"""Reads in the identity configuration file.

    The identity configuration file contains paths to the labeled data
    and a list of stopwords.

    Returns
    -------
    me : path
        The path to the me labeled data.
    notme : path
        The path to the notme labeled data.
    stopwords : list
        A list of stopwords used for the classifier.

    """
    parser = SafeConfigParser()
    parser.read('identity.conf')
    me_data_file = parser.get('Training data', 'me_data_file')
    if os.path.exists(me_data_file):
        me = os.path.abspath(me_data_file)
    notme_data_file = parser.get('Training data', 'notme_data_file')
    if os.path.exists(notme_data_file):
        notme = os.path.abspath(notme_data_file)
    stopwords = parser.get('Stopwords', 'stopwords')
    return me, notme, stopwords


def import_labeled_data():
    r"""Imports the labeled data and runs preprocessing steps.

    Preprocessing is done using the following methods:
    1. regex
    2. space
    3. treebank
    4. whitespace
    5. wordpunct
    6. stanford
    7. sklearn
    Each tokenization algorithm is different and will lead to different
    feature spaces. Including all of the different methods will allow the
    user to investigate the impact of tokenization on the classifier.

    Returns
    -------
    me_df : pandas.DataFrame
        The `me` labeled data in a pandas DataFrame.
    notme_df : pandas.DataFrame
        The `notme` labeled data in a pandas DataFrame.

    """
    me, notme, stopwords = get_config()
    # Import the `me` labeled data.
    me_df = pd.DataFrame()
    with open(me, 'r') as f:
        for line in f:
            raw = line
            regex = regex_tokenizer.tokenize(raw)
            space = space_tokenizer.tokenize(raw)
            treebank = treebank_tokenizer.tokenize(raw)
            whitespace = whitespace_tokenizer.tokenize(raw)
            wordpunct = wordpunct_tokenizer.tokenize(raw)
            stanford = stanford_tokenizer.tokenize(raw)
            cv = CountVectorizer()
            cv.fit_transform([raw])
            sklearn = cv.get_feature_names()
            me_df = me_df.append({
                'raw': raw,
                'regex': regex,
                'space': space,
                'treebank': treebank,
                'whitespace': whitespace,
                'wordpunct': wordpunct,
                'stanford': stanford,
                'sklearn': sklearn,
                'label': 'me'
            }, ignore_index=True)
    # Import the `notme` labeled data.
    notme_df = pd.DataFrame()
    with open(notme, 'r') as f:
        for line in f:
            raw = line
            regex = regex_tokenizer.tokenize(raw)
            space = space_tokenizer.tokenize(raw)
            treebank = treebank_tokenizer.tokenize(raw)
            whitespace = whitespace_tokenizer.tokenize(raw)
            wordpunct = wordpunct_tokenizer.tokenize(raw)
            stanford = stanford_tokenizer.tokenize(raw)
            cv = CountVectorizer()
            cv.fit_transform([raw])
            sklearn = cv.get_feature_names()
            notme_df = me_df.append({
                'raw': raw,
                'regex': regex,
                'space': space,
                'treebank': treebank,
                'whitespace': whitespace,
                'wordpunct': wordpunct,
                'stanford': stanford,
                'sklearn': sklearn,
                'label': 'notme'
            }, ignore_index=True)
    return me_df, notme_df


def split_data(me_df, notme_df, percent=75):
    r"""Splits the labeled data into two sets, a training set and a testing
    set. The returned data is then randomized to prevent bias in the
    classifier.

    Parameters
    ----------
    me_df : pandas.DataFrame
        The `me` labeled data.
    notme_df : pandas.DataFrame
        The `notme` labeled data.
    percent : int, DEFAULT 75
        The amount of data to be put into the training set.

    Returns
    -------
    training_df : pandas.DataFrame
        The training data.
    testing_df : pandas.DataFrame
        The testing data.
    split_idx : dictionary
        The indexes where to split the data on.

    """
    # Split the data up.
    data_split_idx = lambda percent,df: \
                                    int(np.floor((float(percent)/100)*len(df)))
    me_idx = data_split_idx(percent, me_df)
    notme_idx = data_split_idx(percent, notme_df)
    training_df = pd.concat([me_df[:me_idx], notme_df[:notme_idx]],
                            ignore_index=True)
    testing_df = pd.concat([me_df[me_idx:], notme_df[notme_idx:]],
                           ignore_index=True)
    training_df = training_df.reindex(np.random.permutation(training_df.index))
    testing_df = testing_df.reindex(np.random.permutation(testing_df.index))
    split_idx = {'me_idx': me_idx, 'notme_idx': notme_idx}
    return split_idx, training_df, testing_df


def vectorizer(ngram, kind, stopwords=None, tokenizer=None):
    # NOTE:
    #   * Investigate tokenization on the efficiency of the classifier.
    #   * Investigate using stopwords.
    #   * Investigate if case normalization removes too much semantic content.
    #   * Stemming may or may not be useful, and we need to investigate this.
    r"""Generates the sklearn Vectorizer.

    This process will do most of the preprocessing and feature extraction
    steps when dealing with text data. Below is a list of the preprocessing
    and feature extraction techniques employed by the sklearn vectorizer.

    * Tokenization
      Single character tokens are kept, e.g. `I` and `a` are kept in the
      feature space. The default in sklearn is to ignore these characters,
      however, we are attempting to build a classifier that is capable of
      distinguishing the usage of `I` vs `you` or `him` etc so they must
      be kept within the feature space. Punctuation is also kept, which
      may need to be filtered based on the performance of the classifier
      and to reduce the feature space when training.
    * Stopwords
      sklearn will use the supplied stopwords to reduce the feature space,
      however, we must be careful as typical English stopword lists will
      remove words like `me` and `I` etc. If using a stopword list,
      be sure to remove words like `I` or `me` from the list so that
      they are not removed from the feature space.
    * Case normalization
      The vectorizer will normalize case to be lower for all tokens.
      Doing so will remove semantic context within the feature space
      and may not be a good idea, however, it is the simplest process
      to do.
    * Stemming
      sklearn does not feature stemmers or fancy tokenization algorithms.
      NLTK does and you can create your own stemmer if necessary. The
      usage of a stemmer still needs to be investigated.

    The vectorizer will tokenize the input data into
    word tokens that include alphanumeric entries.

    Parameters
    ----------
    ngram : int
        Choice between single word tokens, two-tuple tokens, or three-tuple
        tokens, etc.
    kind : {count, freq, tfidf}, str
        The type of feature extraction to use.
        count: returns simple word counts in a document.
        freq: returns word frequencies in a document.
        tfidf: returns term frequency * inverse document frequencies.
    stopwords : list, DEFAULT None
        A list of stopwords to use.
    tokenizer : object, DEFAULT None
        Can be a function to do fancy tokenization or stemming or
        lemmatization. Things that sklearn does not do.

    Returns
    -------
    An sklearn vectorizer.

    """
    token_pattern = u'(?u)\\b\\w+\\b'
    if kind == 'count':
        return CountVectorizer(token_pattern=token_pattern,
                               ngram_range=(1, ngram),
                               stop_words=stopwords,
                               tokenizer=tokenizer)
    elif kind == 'freq':
        return TfidfVectorizer(token_pattern=token_pattern, use_idf=False,
                               ngram_range=(1, ngram),
                               stop_words=stopwords,
                               tokenizer=tokenizer)
    elif kind == 'tfidf':
        return TfidfVectorizer(token_pattern=token_pattern,
                               ngram_range=(1, ngram),
                               stop_words=stopwords,
                               tokenizer=tokenizer)


def training_vocab(split_idx, me_df, notme_df, ngram):
    r"""Builds pandas dataframes based on `me` and `notme` vocabularies.

    Parameters
    ----------
    split_idx :
    me_df :
    notme_df :

    Returns
    -------
    training_vocabulary_count : pandas.DataFrame
        The training vocabulary dataframe based on word counts.
    training_vocabulary_freq : pandas.DataFrame
        The training vocabulary dataframe based on word frequencies.
    training_vocabulary_tfidf : pandas.DataFrame
        The training vocabulary dataframe based on tfidfs.

    """
    me_idx = split_idx['me_idx']
    notme_idx = split_idx['notme_idx']
    # Look at the me and notme vocabularies.
    training_me_df = me_df[:me_idx]
    training_notme_df = notme_df[:notme_idx]

    # Training me vocabulary counts.
    training_me_vect = vectorizer(ngram=ngram, kind='count')
    training_me_trans = training_me_vect.fit_transform(training_me_df.text)
    training_me_vocab = training_me_vect.vocabulary_
    training_me_dict = {\
        w: pd.Series(training_me_trans.getcol(training_me_vocab[w]).data)
        for w in training_me_vocab}
    training_me_vocab_df = pd.DataFrame(training_me_dict).fillna(0)

    # Training notme vocabulary counts.
    training_notme_vect = vectorizer(ngram=1, kind='count')
    training_notme_trans = training_notme_vect.fit_transform(\
        training_notme_df.text)
    training_notme_vocab = training_notme_vect.vocabulary_
    training_notme_dict = {\
        w: pd.Series(training_notme_trans.getcol(training_notme_vocab[w]).data)
        for w in training_notme_vocab}
    training_notme_vocab_df = pd.DataFrame(training_notme_dict).fillna(0)

    # Training vocabulary counts dataframe.
    training_vocabulary_count = pd.DataFrame(\
            {'me': training_me_vocab_df.sum(axis=0),
             'notme': training_notme_vocab_df.sum(axis=0)}).fillna(0)

    # Training me vocabulary frequencies.
    training_me_vect = vectorizer(ngram=ngram, kind='freq')
    training_me_trans = training_me_vect.fit_transform(training_me_df.text)
    training_me_vocab = training_me_vect.vocabulary_
    training_me_dict = {\
        w: pd.Series(training_me_trans.getcol(training_me_vocab[w]).data)
        for w in training_me_vocab}
    training_me_vocab_df = pd.DataFrame(training_me_dict).fillna(0)

    # Training notme vocabulary frequencies.
    training_notme_vect = vectorizer(ngram=1, kind='freq')
    training_notme_trans = training_notme_vect.fit_transform(\
        training_notme_df.text)
    training_notme_vocab = training_notme_vect.vocabulary_
    training_notme_dict = {\
        w: pd.Series(training_notme_trans.getcol(training_notme_vocab[w]).data)
        for w in training_notme_vocab}
    training_notme_vocab_df = pd.DataFrame(training_notme_dict).fillna(0)

    # Training vocabulary frequency dataframe.
    training_vocabulary_freq = pd.DataFrame(\
            {'me': training_me_vocab_df.sum(axis=0),
             'notme': training_notme_vocab_df.sum(axis=0)}).fillna(0)

    # Training me vocabulary tfidf.
    training_me_vect = vectorizer(ngram=ngram, kind='tfidf')
    training_me_trans = training_me_vect.fit_transform(training_me_df.text)
    training_me_vocab = training_me_vect.vocabulary_
    training_me_dict = {\
        w: pd.Series(training_me_trans.getcol(training_me_vocab[w]).data)
        for w in training_me_vocab}
    training_me_vocab_df = pd.DataFrame(training_me_dict).fillna(0)

    # Training notme vocabulary tfidf.
    training_notme_vect = vectorizer(ngram=ngram, kind='tfidf')
    training_notme_trans = training_notme_vect.fit_transform(\
        training_notme_df.text)
    training_notme_vocab = training_notme_vect.vocabulary_
    training_notme_dict = {\
        w: pd.Series(training_notme_trans.getcol(training_notme_vocab[w]).data)
        for w in training_notme_vocab}
    training_notme_vocab_df = pd.DataFrame(training_notme_dict).fillna(0)
    training_vocabulary_tfidf = pd.DataFrame(\
        {'me': training_me_vocab_df.sum(axis=0),
         'notme': training_notme_vocab_df.sum(axis=0)}).fillna(0)

    # Vocabulary dataframes.
    training_vocabulary_tfidf = pd.DataFrame(\
        {'me': training_me_vocab_df.sum(axis=0),
         'notme': training_notme_vocab_df.sum(axis=0)}).fillna(0)

    return training_vocabulary_count, training_vocabulary_freq, \
           training_vocabulary_tfidf


def plot_vocab(df, y1, y2, title, xlabel, ylabel):
    r"""Plots the given dataframe with regards to vocabulary.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to plot.
    title : string
        The title of the plot.
    xlabel : string
        The x axis label of the plot.
    ylabel : string
        The y axis label of the plot.

    Returns
    -------
    A plot.

    """
    description = list(df.index)
    x = range(len(description))
    me_source = ColumnDataSource(data=dict(x=x,
                                           y=y1,
                                           desc=description,
                                 ))
    notme_source = ColumnDataSource(data=dict(x=x,
                                              y=y2,
                                              desc=description,
                                    ))
    hover = HoverTool(tooltips=[("", "@desc"), ])
    p = figure(plot_width=950, plot_height=500, tools=[hover,
                                                       WheelZoomTool(),
                                                       PanTool(),
                                                       ResetTool(),
                                                       BoxZoomTool()],
               title=title,
               x_axis_label=xlabel,
               y_axis_label=ylabel)

    p.circle('x', 'y', size=6, source=me_source, fill_color='green',
             line_color='green', line_alpha=0.3, fill_alpha=0.3)
    p.line('x', 'y', source=me_source, line_color='green',
           line_alpha=0.3, legend='Me')
    p.circle('x', 'y', size=6, source=notme_source, fill_color='red',
             line_color='red', line_alpha=0.3, fill_alpha=0.3)
    p.line('x', 'y', source=notme_source, line_color='red',
           line_alpha=0.3, legend='Not Me')
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    show(p)


def compare_classifiers():
    r"""
    """




# # Unigram counts.
# unigram_count_vect = preprocess(unigram=True, kind='count')
# unigram_count_trans = unigram_count_vect.fit_transform(training_df.text)
# unigram_count_vocab = unigram_count_vect.vocabulary_
# unigram_count_dict = {\
#     w: pd.Series(unigram_count_trans.getcol(unigram_count_vocab[w]).data)
#     for w in unigram_count_vocab}
# unigram_count_df = pd.DataFrame(unigram_count_dict).fillna(0)

# # Bigram counts.
# bigram_count_vect = preprocess(bigram=True, kind='count')
# bigram_count_trans = bigram_count_vect.fit_transform(training_df.text)
# bigram_count_vocab = bigram_count_vect.vocabulary_
# bigram_count_dict = {\
#     w: pd.Series(bigram_count_trans.getcol(bigram_count_vocab[w]).data)
#     for w in bigram_count_vocab}
# bigram_count_df = pd.DataFrame(bigram_count_dict).fillna(0)

# # Unigram frequencies.
# unigram_freq_vect = preprocess(unigram=True, kind='freq')
# unigram_freq_trans = unigram_freq_vect.fit_transform(training_df.text)
# unigram_freq_vocab = unigram_freq_vect.vocabulary_
# unigram_freq_dict = {\
#     w: pd.Series(unigram_freq_trans.getcol(unigram_freq_vocab[w]).data)
#     for w in unigram_freq_vocab}
# unigram_freq_df = pd.DataFrame(unigram_freq_dict).fillna(0)

# # Bigram frequencies.
# bigram_freq_vect = preprocess(bigram=True, kind='freq')
# bigram_freq_trans = bigram_freq_vect.fit_transform(training_df.text)
# bigram_freq_vocab = bigram_freq_vect.vocabulary_
# bigram_freq_dict = {\
#     w: pd.Series(bigram_freq_trans.getcol(bigram_freq_vocab[w]).data)
#     for w in bigram_freq_vocab}
# bigram_freq_df = pd.DataFrame(bigram_freq_dict).fillna(0)

# # Unigram tfidf.
# unigram_tfidf_vect = preprocess(unigram=True, kind='tfidf')
# unigram_tfidf_trans = unigram_tfidf_vect.fit_transform(training_df.text)
# unigram_tfidf_vocab = unigram_tfidf_vect.vocabulary_
# unigram_tfidf_dict = {\
#     w: pd.Series(unigram_tfidf_trans.getcol(unigram_tfidf_vocab[w]).data)
#     for w in unigram_tfidf_vocab}
# unigram_tfidf_df = pd.DataFrame(unigram_tfidf_dict).fillna(0)

# # Bigram tfidf.
# bigram_tfidf_vect = preprocess(bigram=True, kind='tfidf')
# bigram_tfidf_trans = bigram_tfidf_vect.fit_transform(training_df.text)
# bigram_tfidf_vocab = bigram_tfidf_vect.vocabulary_
# bigram_tfidf_dict = {\
#     w: pd.Series(bigram_tfidf_trans.getcol(bigram_tfidf_vocab[w]).data)
#     for w in bigram_tfidf_vocab}
# bigram_tfidf_df = pd.DataFrame(bigram_tfidf_dict).fillna(0)
