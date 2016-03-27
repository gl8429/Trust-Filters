import numpy
import pandas
import requests
import urlparse
import json
# import pdb
#import Stopwords
import blog_scraper
import blog_game
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import KFold
# from sklearn.grid_search import GridSearchCV

stpwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'an', 'and', 'any', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'have', 'having', 'here', 'how', 'if', 'in', 'into', 'just', 'more', 'most', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'out', 'over', 'own', 's', 'same', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'then', 'there', 'these', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with']



class TextClassifier(object):

    def __init__(self):
        """preprocesses data, selects pipeline, runs classification"""

        self.data = self.build_dataframes()
        self.pipeline, self.fold = self.choose_pipeline()
        self.cross_validate(self.pipeline, self.fold)
        # self.print_results()
        # grid search stuff?

    def build_dataframes(self):
        """loads me/not me training data into pandas dataframe, and
        preprocesses for use in classifier"""

        # load 'me' data
        self.me_df = pandas.DataFrame(columns=['text', 'class'])
        with open('ME.txt', 'rb') as file:
            for line in file.readlines():
                row = pandas.Series({'text': line, 'class': 'me'})
                self.me_df = self.me_df.append(row, ignore_index=True)

        # load 'not_me' data
        self.not_df = pandas.DataFrame(columns=['text', 'class'])
        with open('NOT_ME.txt', 'rb') as file:
            for line in file.readlines():
                row = pandas.Series({'text': line, 'class': 'not'})
                self.not_df = self.not_df.append(row, ignore_index=True)

        # combine and shuffle dataframes
        df = pandas.concat([self.me_df, self.not_df], ignore_index=True)
        df = df.reindex(numpy.random.permutation(df.index))
        return df

    def choose_pipeline(self):
        """prints out pipeline options for user to choose from, returns
        selected pipeline"""

        names = []
        options = {}
        names.append('naiive_bayes')
        names.append('naiive_bayes_plus_2grams')
        names.append('svm')
        names.append('naiive_bayes_plus_tfidf')
        names.append('svm_plus_tfidf')
        names.append('bernouli')
        names.append('svm_plus_stopwords')
        names.append('svm_plus_everything')

        print "Classifier choices:\n"
        for index, name in enumerate(names):
            print "%r - " % index + name
            options[index] = "_" + name

        choice = raw_input("\nPlease choose a classifier by entering the " +
                           "appropriate number: ")
        pipeline = getattr(self, options[int(choice)])()
        fold = raw_input("Please choose a fold for cross-validation: ")
        return pipeline, int(fold)

    def cross_validate(self, pipeline, fold=6):
        """performs k-fold cross-validation using given pipeline"""

        k_fold = KFold(n=len(self.data), n_folds=fold, shuffle=True,
                       indices=False)
        scores = []
        print ""
        print pipeline.get_params()['clf']
        print ""

        for train_indices, test_indices in k_fold:

            train_text = list(self.data[train_indices]['text'])
            train_class = list(self.data[train_indices]['class'])

            test_text = list(self.data[test_indices]['text'])
            test_class = list(self.data[test_indices]['class'])

            pipeline.fit(train_text, train_class)
            score = pipeline.score(test_text, test_class)
            print score
            scores.append(score)

        score = numpy.sum(scores) / len(scores)
        pipeline.fit(list(self.data['text']), list(self.data['class']))
        print "\nCross-validation average: " + str(score)

    def classify_list(self, list_to_classify):
        """uses classifier to classify a list of strings"""

        print self.pipeline.predict(list_to_classify).tolist()

    def classify_post(self, post_path):
        """uses classifier to classify a blog post"""

        paragraph_list = blog_scraper.get_paragraphs(post_path)
        predictions = self.pipeline.predict(paragraph_list).tolist()

        for text, prediction in zip(paragraph_list, predictions):
            print "Text: " + text
            print "\033[1;34m" + "Classification: " + prediction + "\033[0m\n"
        # TODO: average predictions for a measure of the classification.

    # pipeline options -  all have token_pattern for single-letter words
    def _naiive_bayes(self):
        return Pipeline([('cv', CountVectorizer(token_pattern='\\b\\w+\\b')),
                         ('clf', MultinomialNB())])

    def _naiive_bayes_plus_2grams(self):
        return Pipeline([('cv', CountVectorizer(token_pattern='\\b\\w+\\b',
                          ngram_range=(1, 2))), ('clf', MultinomialNB())])

    def _svm(self):
        return Pipeline([('cv', CountVectorizer(token_pattern='\\b\\w+\\b')),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, n_iter=5))])

    def _naiive_bayes_plus_tfidf(self):
        return Pipeline([('cv', CountVectorizer(token_pattern='\\b\\w+\\b')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])

    def _svm_plus_tfidf(self):
        return Pipeline([('cv', CountVectorizer(token_pattern='\\b\\w+\\b')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, n_iter=5))])

    def _bernouli(self):
        return Pipeline([('cv', CountVectorizer(token_pattern='\\b\\w+\\b')),
                         ('clf', BernoulliNB(binarize=0.0))])

    def _svm_plus_stopwords(self):
        return Pipeline([('cv', CountVectorizer(token_pattern='\\b\\w+\\b',
                          stop_words=stpwords)),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, n_iter=5))])

    def _svm_plus_everything(self):
        return Pipeline([('cv', CountVectorizer(token_pattern='\\b\\w+\\b',
                          stop_words=stpwords,
                          ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, n_iter=5))])

    def get_vocab(self, num=10, stopwords=True):
        """gets 10 most used words for each category"""

        categories = [self.me_df, self.not_df]

        for category in categories:
            # pdb.set_trace()
            print ""
            if stopwords:
                cv = CountVectorizer(token_pattern='\\b\\w+\\b',
                                     stop_words=stpwords)
            else:
                cv = CountVectorizer(token_pattern='\\b\\w+\\b')
            counts = cv.fit_transform(list(category['text']))
            counts = counts.toarray()
            vocab = cv.get_feature_names()
            vocab = numpy.array(vocab)
            totals = counts.sum(axis=0)
            indices = numpy.argsort(totals)[num * -1:]

            if category['class'][0] == 'me':
                print "ME blogs:"
            else:
                print "NOT ME blogs:"

            for i in reversed(indices):
                print vocab[i]

    def improve_classifier(self):
        """uses reinforcement learning concepts to improve classifier"""

        blog_game.user_interface()
        self.data = self.build_dataframes()
        self.cross_validate(self.pipeline, self.fold)


def save_blog_content(site, post_id):
    """returns path to blog content saved in json format"""

    # get json data from site
    base_url = 'https://public-api.wordpress.com/rest/v1/sites/'
    url = lambda x, y: urlparse.urljoin(base_url, x + '/posts/' + y)
    r = requests.get(url(site, post_id))
    data = r.json()
    data['comments'] = ''
    r.close()

    # save json data and return path
    post_path = site + '_' + post_id + '.json'
    json.dump(data, open(post_path, 'wb'), indent=4)

    return post_path


# INCOMPLETE -- the idea is to call the gridsearch function, and use the best
# parameters it finds to build an optimal pipeline.

# def build_pipeline():
#     """build best pipeline using gridsearch"""

#     parameters = {'cv__ngram_range': [(1, 1), (1, 2)],
#                   'tfidf__use_idf': (True, False),
#                   'classifier__alpha': (1e-2, 1e-3)}
#     best_parameters = grid_search(parameters)


# def grid_search(parameters):
#     """find best parameters using gridsearch"""

#     pipeline = Pipeline([('cv', CountVectorizer(token_pattern='\\b\\w+\\b')),
#                          ('tfidf', TfidfTransformer()),
#                          ('classifier', SGDClassifier(loss='hinge',
#                           penalty='l2', alpha=1e-3, n_iter=5))])
#     gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
#     gs_clf = gs_clf.fit(list(data['text'][:300]), list(data['class'][:300]))
#     best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])

#     for param_name in sorted(parameters.keys()):
#         print("%s: %r" % (param_name, best_parameters[param_name]))

#     return... what?
