import tc_constructor
import cPickle
import blog_scraper
import numpy
from sklearn.feature_extraction.text import CountVectorizer


class TextClassifier(object):

    def __init__(self, clf_name='', fold=6,
                 pkl_clf_file='my_dumped_classifier.pkl', load=False):
        """preprocesses data, selects pipeline, runs classification"""

        # if parameters are supplied, create new classifier
        if clf_name != '':
            self.tc = tc_constructor.TextClassifierObject(clf_name=clf_name,
                                                          fold=fold)

        # if no parameters and load=True
        elif load:
            # try to access file
            try:
                self.tc = self.load_classifier(pkl_clf_file)
            # if no file, prompt user for input
            except:
                self.tc = tc_constructor.TextClassifierObject()

        # if no parameters and no file, prompt user for input
        else:
            self.tc = tc_constructor.TextClassifierObject()

    def save_classifier(self, pkl_clf_file='my_dumped_classifier.pkl'):
        """uses pickle (object serialization) to save classifier"""

        with open(pkl_clf_file, 'wb') as fid:
            cPickle.dump(self.tc, fid)

    def load_classifier(self, pkl_clf_file):
        """returns previously pickled classifier"""

        with open(pkl_clf_file, 'rb') as fid:
            tc_loaded = cPickle.load(fid)

        return tc_loaded

    def classify_list(self, list_to_classify):
        """uses classifier to classify a list of strings"""

        print self.tc.pipeline.predict(list_to_classify).tolist()

    def classify_post(self, post_path):
        """uses classifier to classify a blog post"""

        paragraph_list = blog_scraper.get_paragraphs(post_path)
        predictions = self.tc.pipeline.predict(paragraph_list).tolist()

        for text, prediction in zip(paragraph_list, predictions):
            print "Text: " + text
            print "\033[1;34m" + "Classification: " + prediction + "\033[0m\n"
        # TODO: average predictions for a measure of the classification.

    def get_vocab(self, num=10, stopwords=True):
        """gets 10 most used words for each category"""

        categories = [self.tc.me_df, self.tc.not_df]

        for category in categories:
            print ""
            sw = tc_constructor.Stopwords.my_stopwords
            if stopwords:
                cv = CountVectorizer(token_pattern='\\b\\w+\\b',
                                     stop_words=sw)
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
