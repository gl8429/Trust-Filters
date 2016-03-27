# -*- coding: utf-8 -*-

r"""
"""

import os
import json
import logging

import langdetect
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.internals import find_jars_within_path
from nltk.tokenize import StanfordTokenizer

logger = logging.getLogger(__name__)


class FetchTestData(object):

    def __init__(self, basedir, lower=100, upper=200, dtype='tokens'):
        r"""Fetches test data from the remote server.

        Parameters
        ----------
        basedir : string
            The directory where the data exists.
        lower : integer
            DEFAULT: 100
            A reasonable lower bound to test data with.
        upper : integer
            DEFAULT: 200
            A reasonable upper bound to test data with.
        dtype : string
            **Default** tokens
            Can be one of the following:
                * raw
                * tokens
                * flu

        Example
        -------
        >>> from bioevent.trust import utils
        >>> basepath = '/vagrant/bsve_datastore/blogs/wordpress/done/'
        >>> fetch = utils.FetchTestData(basepath)
        >>> # Return a dictionary of tokens.
        >>> data = fetch.tokens()

        """
        self._basedir = basedir
        self._lower = lower
        self._upper = upper
        self._dtype = dtype
        self._blogdirs = [os.path.join(self._basedir, folder)
                          for folder in os.listdir(self._basedir)
                          if os.path.exists(os.path.join(self._basedir,
                                            folder))]

    def tokens(self):
        r"""Retrieves tokenized data.

        Returns
        -------
        test_data : dictionary
            A dictionary where the keys are the name of the file and the
            values are the tokens from that file.

        """
        self._dtype = 'tokens'
        test_data = self._get_sample()
        return test_data

    def raw(self):
        r"""Retrieves raw data.

        Returns
        -------
        test_data : dictionary
            A dictionary where the keys are the names of the files and the
            values are the raw text strings from that file.

        """
        self._dtype = 'raw'
        test_data = self._get_sample()
        return test_data

    def flu(self):
        r"""Retrieves data containing flu words.

        Returns
        -------
        test_data : dictionary
            A dictionary where the keys are the names of the files and the
            values are the tokens from that file.

        """
        self._dtype = 'flu'
        test_data = self._get_sample()
        return test_data

    def _get_sample(self):
        r"""Returns a corpus with a bounded amount of documents in it for
        testing.

        Returns
        -------
        corpus : dictionary
            A dictionary of documents where the keys are file names and
            values are the tokens or strings from that file.

        """
        s = np.random.choice(self._blogdirs, size=1)
        blog_posts = os.path.join(s.item(0), 'posts')
        nbr_posts = len(os.listdir(blog_posts))
        if nbr_posts >= self._lower and nbr_posts <= self._upper:
            print s.item(0)
            if self._dtype == 'tokens':
                data = os.path.join(s.item(0), 'features/tokens_lower.json')
            elif self._dtype == 'raw':
                data = os.path.join(s.item(0), 'raw.json')
            elif self._dtype == 'flu':
                data = os.path.join(s.item(0), 'flu_posts_tokens.json')
            try:
                with open(data, 'rb') as f:
                    corpus = json.load(f)
                return corpus
            except:
                return self._get_sample()
        if nbr_posts < self._lower:
            return self._get_sample()
        if nbr_posts > self._upper:
            return self._get_sample()


class CleanTokens(object):

    def __init__(self):
        r"""The methods attached to this class will clean the given tokens
        from punctuation or stopwords and can return tokens that are only
        found to be English.

        Example
        -------
        >>> from bioevent.trust import utils
        >>> clean = utils.CleanTokens()
        >>> tokens = ['This', 'is', 'a', 'sentence', '.']
        >>> no_punctuation = clean.punctuation(tokens)
        >>> no_punctuation
        ['This', 'is', 'a', 'sentence']

        """
        self._stopwords = stopwords.words('english')

    def lowercase(self, tokens):
        r"""Normalizes the case of the supplied tokens list.

        Parameters
        ----------
        tokens : list
            A list of token strings.

        Returns
        -------
        tokens : list
            A list of tokens that are now all lowercase.

        """
        tokens = [token.lower() for token in tokens]
        return tokens

    def punctuation(self, tokens):
        r"""Removes all punctuation from the supplied list of tokens.

        Parameters
        ----------
        tokens : list
            A list of token strings.

        Returns
        -------
        tokens : list
            A cleaned list of token strings.

        """
        tokens = [token for token in tokens if token.isalpha()]
        return tokens

    def stopwords(self, tokens):
        r"""Removes all stopwords from the supplied list of tokens.

        Parameters
        ----------
        tokens : list
            A list of token strings.

        Returns
        -------
        tokens : list
            A cleaned list of token strings.

        """
        tokens = [token for token in tokens if token not in self._stopwords]
        return tokens

    def english(self, tokens):
        r"""If the supplied list of tokens are English, then this method will
        return the list of tokens back. If the tokens are not English, then
        it will return an empty list.

        Parameters
        ----------
        tokens : list
            A list of token strings.

        Returns
        -------
        tokens : list
            A cleaned list of token strings.

        """
        if tokens:
            lang = langdetect.detect(' '.join(tokens))
            if lang == 'en':
                return tokens
        else:
            return []

    def normalize(self, tokens):
        r"""Normalizes the given list of tokens so that there contains no
        punctuation, all tokens are lowercase, all stopwords have been removed,
        and the tokens are English words.

        Parameters
        ----------
        tokens : list
            A list of token strings.

        Returns
        -------
        tokens : list
            A cleaned list of token strings. If the tokens are not English,
            then this will return an empty list.

        """
        tokens = self.english(tokens)
        if tokens:
            tokens = self.punctuation(tokens)
            tokens = self.lowercase(tokens)
            tokens = self.stopwords(tokens)
            return tokens
        else:
            return []


class FeatureExtraction(object):

    def __init__(self):
        r"""Extracts token features from the supplied corpus.

        Parameters
        ----------
        corpus : dictionary
            A dictionary of the form:

            {
                'key1': ['token1', 'token2', 'token3', ...],
                'key2': ['token1', ...],
                'key3': [],
                ...
            },

            where the dictionary keys are pointers to the file in which the
            tokens (values) originate.

        Example
        -------
        >>> from bioevent.trust import utils
        >>> basepath = '/vagrant/bsve_datastore/blogs/wordpress/done/'
        >>> fetch = utils.FetchTestData(basepath)
        >>> data = fetch.tokens()
        >>> extract = utils.FeatureExtraction()
        >>> counts = extract.counts(data)
        >>> extract.word_index
        {
            'word': 4,
            'the': 40,
            'flu': 3,
            ...
        }
        >>> counts.todense()
        np.array([[0, 0, 3, ...],
                  [3, 0, 0, ...],
                  ...])

        """
        self.word_index = {}

    def token_counts(self, corpus):
        r"""Returns a compressed sparse row matrix with the counts of words in
        each document.

        Parameters
        ----------
        corpus : dictionary
            A dictionary with the corpus, where the keys are the name of the
            document and the values are a list of the tokens in those
            documents.

        Returns
        -------
        counts : scipy.sparse.csr_matrix
            A Compressed Sparse Row matrix.

        """
        indptr = [0]
        indices = []
        matrix_data = []
        for blog,token_list in corpus.items():
            for token in token_list:
                index = self.word_index.setdefault(token, len(self.word_index))
                indices.append(index)
            indptr.append(len(indices))
        matrix_data = np.ones(len(indices))
        counts = sp.csr_matrix((matrix_data, indices, indptr),
                               shape=(len(corpus), len(self.word_index)),
                               dtype=np.float64)
        counts.sum_duplicates()
        return counts

    def token_frequencies(self, corpus):
        r"""Returns a compressed sparse row matrix with the frequencies of
        tokens in each document.

        Parameters
        ----------
        corpus : dictionary
            A dictionary with the corpus, where the keys are the name of the
            document and the values are a list of the tokens in those
            documents.

        Returns
        -------
        frequencies : scipy.sparse.csr_matrix
            A Compressed Sparse Row matrix.

        """
        counts = self.token_counts(corpus)
        indptr = counts.indptr
        indices = counts.indices
        matrix_data = counts.data
        doc_lengths = [doc.toarray().sum() for doc in counts]
        data_indices = zip(indptr, indptr[1:])
        for (i, (start, end)) in enumerate(data_indices):
            if i == len(data_indices):
                end = end + 1
            matrix_data[start:end] /= doc_lengths[i]
        frequencies = sp.csr_matrix((matrix_data, indices, indptr),
                                    shape=(len(corpus), len(self.word_index)),
                                    dtype=np.float64)
        frequencies.sum_duplicates()
        return frequencies

    def token_tfidf(self, corpus):
        r"""Returns a compressed sparse row matrix with the tfidfs of
        tokens in each document.

        Parameters
        ----------
        corpus : dictionary
            A dictionary with the corpus, where the keys are the name of the
            document and the values are a list of the tokens in those
            documents.

        Returns
        -------
        tfidf : scipy.sparse.csr_matrix
            A Compressed Sparse Row matrix.

        """
        msg = 'This is broken. Do not use.'
        raise IOError(msg)
        #frequency = self.token_frequencies(corpus)
        #indptr = frequency.indptr
        #indices = frequency.indices
        #matrix_data = frequency.data
        #data_indices = zip(indptr, indptr[1:])
        #N = float(len(corpus))
        #df = frequency.getnnz(axis=0)
        #for (i, (start, end)) in enumerate(data_indices):
        #    if i == len(data_indices):
        #        end = end + 1
        #    for j in indices[start:end]:
        #        idf = np.log10(N/df[j])
        #        matrix_data[j] *= idf
        #tfidf = sp.csr_matrix((matrix_data, indices, indptr),
        #                       shape=(len(corpus), len(self.word_index)),
        #                       dtype=np.float64)
        #return tfidf


class Tokenizer(object):

    def __init__(self, prose):
        r"""Fixes a bug in NLTK so that the Stanford Tokenizer knows where
        all the paths to the jars it needs to tokenize are.

        Parameters
        ----------
        prose : string
            The string to tokenize

        Returns
        -------
        A list of tokens using the Stanford Natural Language Parser.

        Example
        -------
        >>> import utils
        >>> mytokenizer = utils.Tokenizer('This is a sentence.')
        >>> mytokenizer.tokenize
        ['This', 'is', 'a', 'sentence', '.']

        """
        self.tokenize = prose

    @property
    def tokenize(self):
        return self.__tokenize

    @tokenize.setter
    def tokenize(self, s):
        # Explanations of the options can be found here.
        # http://nlp.stanford.edu/software/tokenizer.shtml
        options = {'americanize': True,
                   'normalizeSpace': True,
                   'normalizeAmpersandEntity': True,
                   'normalizeFractions': True,
                   'normalizeParentheses': True,
                   'normalizeOtherBrackets': True,
                   'asciiQuotes': True,
                   'unicodeQuotes': True,
                   'ptb3Ellipsis': True,
                   'unicodeEllipsis': True,
                   'ptb3Dashes': True}
        # This is a hack to fix tokenization.
        # https://github.com/nltk/nltk/issues/1239
        stanford_dir = '/vagrant/stanford_nlp'
        jar = os.path.join(stanford_dir, 'stanford-parser.jar')
        stanford_jars = find_jars_within_path(stanford_dir)
        tokenizer = StanfordTokenizer(path_to_jar=jar,
                                      java_options='-Xmx4G',
                                      options=options)
        tokenizer._stanford_jar = ':'.join(stanford_jars)
        self.__tokenize = tokenizer.tokenize(s)
