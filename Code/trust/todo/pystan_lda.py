import os
import json
from collections import OrderedDict, Mapping, defaultdict, Counter
from array import array
from copy import deepcopy
import six

import numpy as np
import pystan
import utils








def sparse2dense(sparse_vector):
    r"""
    """
    dense_vector = sparse_vector.todense()
    shape = dense_vector.shape
    dense_vector = np.array(dense_vector).reshape(shape[1])
    return dense_vector


lda_model = """
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Latent Dirichlet Allocation model referenced from the Stan Manual ver 2.8.0
# pages 148--149.
# http://mc-stan.org/documentation/
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data {
    # Measured data.
    int<lower=1> V;                  # Number of distinct words in the corpus.
    int<lower=1> D;                  # Number of documents in the corpus.
    int<lower=1> N;                  # Number of words in the corpus.
    int<lower=1, upper=V> W[N];      # Word n in document d.
    int<lower=1, upper=D> doc[N];    # Document ID for word n.

    # Model input.
    int<lower=2> T;                  # Number of topics to run the model with.

    # Model hyperparameters.
    vector<lower=0>[T] alpha;        # Topic-document distribution prior.
    vector<lower=0>[V] beta;         # Topic-word distribution prior.
}
parameters {
    simplex[T] theta[D];             # Topic-document distribution.
    simplex[V] phi[T];               # Topic-word distribution.
}
model {
    # Priors.
    for (d in 1:D)
        theta[d] ~ dirichlet(alpha);
    for (t in 1:T)
        phi[t] ~ dirichlet(beta);
    # Likelihood.
    for (n in 1:N) {
        real gamma[T];
        for (t in 1:T)
            gamma[t] <- log(theta[doc[n],t]) + log(phi[t,W[n]]);
        increment_log_prob(log_sum_exp(gamma));
    }
    # for (d in 1:D) {
    #     for (n in doc[d,1]:doc[d,2]) {
    #         real gamma[T];
    #         for (t in 1:T)
    #             gamma[t] <- log(theta[d,t]) + log(phi[t,W[n]]);
    #         increment_log_prob(log_sum_exp(gamma));
    #     }
    # }
}
"""


switch_model = """
data {
    int<lower=1> V;
    int<lower=1> D;
    int<lower=1> N;
    int<lower=0,upper=V> W[N];
    int<lower=1,upper=N> doc[D,2];
    int<lower=2> T;
    vector<lower=0>[T] alpha;
    vector<lower=0>[V] beta1;
    vector<lower=0>[V] beta2;
    vector[2] xi;
}
parameters {
    simplex[T] theta[D];
    simplex[V] phi[T];
    simplex[V] phi_ne[T];
    simplex[2] psi[T];

}
model {
    for (d in 1:D)
        theta[d] ~ dirichlet(alpha);
    for (t in 1:T) {
        phi[t] ~ dirichlet(beta1);
        phi_ne[t] ~ dirichlet(beta2);
        psi[t] ~ beta(1,1);
    }
    for (d in 1:D) {
        for (n in doc[d,1]:doc[d,2]) {
            real gamma[T];
            for (t in 1:T) {
                int x[2];
                x ~ multinomial(psi[t]);
                if (x[1] == 1) {
                    gamma[t] <- log(theta[d,t]) + log(phi[t,W[n]]);
                } else {
                    gamma[t] <- log(theta[d,t]) + log(phi_ne[t,W[n]]);
                }
            }
            increment_log_prob(log_sum_exp(gamma));
        }
    }
}
"""

coclustering_model = """
data {
    # Data inputs.
    int<lower=1> V;  # Vocabulary total.
    int<lower=2> D;  # Document total.
    int<lower=V> N;  # Word total.
    int<lower=1> A;  # Author total.
int<lower=1, upper=V> W[N];      # Word n in document d.
int<lower=1, upper=N> doc[D,2];  # Document ID for word n.

    # Model inputs.
    int<lower=1> T;  # Topic total.

    # Hyperpriors
    vector<lower=0>[T] alpha;  # Author-topic hyperparameter.
    vector<lower=0>[T] gamma;  # Document-topic hyperparameter.
}
parameters {

}
model {

}
"""


def generate_model_data(feature_corpus, t):
    r"""Returns the data model for the stan model.

    Parameters
    ----------
    feature_corpus : dictionary
        The input feature corpus.
    t : integer
        The number of topics to model.

    Returns
    -------
    model_data : dictionary
        Returns a simple data model for an LDA.

    """
    T = t
    V = feature_corpus.shape[1]
    D = feature_corpus.shape[0]
    N = len(feature_corpus.data)
    # NOTE
    # The word vector and the docs vector will fail if they are zero indexed.
    # This is a mapping of vocabulary indices to the word in a document.
    W = [idx+1 for doc in feature_corpus for idx in doc.indices]
    # Document offset ID. This is a vector of tuples that indicate when a
    # document starts and finishes with respect to the word index.
    docs = [i+1 for i,doc in enumerate(feature_corpus) for idx in doc.indices]
    # docs = map(list, zip(feature_corpus.indptr, feature_corpus.indptr[1:]))
    # docs[0][0] = 1
    #  docs = []
    #  total_word_count = 0
    #  for i,doc in enumerate(feature_corpus):
    #      if i == 0:
    #          docs.append([total_word_count, int(doc.sum()) - 1])
    #          total_word_count += int(doc.sum())
    #      else:
    #          old = total_word_count
    #          total_word_count += int(doc.sum())
    #          docs.append([old, total_word_count - 1])
    # Hyper-priors.
    alpha = np.repeat(0.8, T)
    beta = np.repeat(0.2, V)
    model_data = {'T': T,
                  'V': V,
                  'D': D,
                  'N': N,
                  'W': W,
                  'doc': docs,
                  'alpha': alpha,
                  'beta': beta}
    return model_data

#  base_path = '/vagrant/bsve_datastore/blogs/wordpress/done/'
#  fetch = utils.FetchTestData(base_path)
#  raw_data = fetch.tokens()
#  word_tfidfs = extract_features(raw_data, 'tfidf')
#  V = word_tfidfs.shape[1]
#  D = word_tfidfs.shape[0]
#  N = len(word_tfidfs.data)
#  W = gen1dvec(word_tfidfs)[0]
#  docs = map(list, zip(word_tfidfs.indptr+1, word_tfidfs.indptr[1:]))
#  T = 40
#  alpha = np.repeat(0.8, T)
#  beta = np.repeat(0.2, V)

#  lda_data = {'V': V,
#              'D': D,
#              'N': N,
#              'W': W,
#              'doc': docs,
#              'T': T,
#              'alpha': alpha,
#              'beta': beta}

#  fit = pystan.stan(model_code=lda_model, data=lda_data)
#  print fit
