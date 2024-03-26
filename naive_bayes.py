import os, math
from collections import Counter

class NaiveBayesClassifier:
    """Code for a bag-of-words Naive Bayes classifier.
    """
    
    def __init__(self, train_dir='haiti/train', REMOVE_STOPWORDS=False):
        self.REMOVE_STOPWORDS = REMOVE_STOPWORDS
        self.stopwords = set([l.strip() for l in open('english.stop')])
        self.classes = os.listdir(train_dir)
        self.train_data = {c: os.path.join(train_dir, c) for c in self.classes}
        self.vocabulary = set([])
        self.logprior = {}
        self.loglikelihood = {} # keys should be tuples in the form (w, c)

        self.n_doc = 0
        self.bigdoc = {}
        
    def train(self):
        """Train the Naive Bayes classification model, following the pseudocode for
        training given in Figure 4.2 of SLP Chapter 4. 

        Note that self.train_data contains the paths to training data files. 
        To get all the documents for a given training class c in a list, you can use:
            c_docs = open(self.train_data[c]).readlines()

        Like in A2, you can assume they are pre-tokenized so you can get words with
        simply `words = doc.split()`

        Remember to account for whether the self.REMOVE_STOPWORDS flag is set or not;
        if it is True then the stopwords in self.stopwords should be removed whenever
        they appear.

        When converting from the pseudocode, consider how many loops over the data you
        will need to properly estimate the parameters of the model, and what intermediary
        variables you will need in this function to store the results of your computations.

        Parameters
        ----------
        None (reads training data from self.train_data)
        
        Returns
        -------
        None (updates class attributes self.vocabulary, self.logprior, self.loglikelihood)
        """
        # >>> YOUR ANSWER HERE
        #make c_docs, self.n_doc, n_c
        n_c = {}
        
        for c in self.classes:
            c_docs = open(self.train_data[c]).readlines() #all docs for class c in list
            self.n_doc += len(c_docs) #number of docs
            n_c[c] = len(c_docs) #number of docs for the class

           #bigdoc#
            self.bigdoc[c] = []
            for document in c_docs:
                self.bigdoc[c] += document.split()
                #remove stopwords#
                for word in self.bigdoc[c]:
                    if self.REMOVE_STOPWORDS:
                        if word not in self.stopwords:
                           self.vocabulary.add(word)
                        if word in self.stopwords:
                            self.bigdoc[c].remove(word)
                    else:
                        self.vocabulary.add(word)
            
        #log prior
        for c in self.classes:
            self.logprior[c]=math.log(n_c[c]/self.n_doc) #log prior calculation

        #num of word w occurances in bigdoc 
            for word in self.vocabulary:
                word_count = self.bigdoc[c].count(word)
                numerator = word_count + 1
                denominator = len(self.bigdoc[c]) + len(self.vocabulary)
                #calculate loglikelihood
                self.loglikelihood[(word,c)] = math.log(numerator/denominator)

        # >>> END YOUR ANSWER
        
    def score(self, doc, c):
        """Return the log-probability of a given document for a given class,
        using the trained Naive Bayes classifier. 

        This is analogous to the inside of the for loop in the TestNaiveBayes
        pseudocode in Figure 4.2, SLP Chapter 4.

        Parameters
        ----------
        doc : str
            The text of a document to score.
        c : str
            The name of the class to score it against.

        Returns
        -------
        float
            The log-probability of the document under the model for class c.
        """        
        # >>> YOUR ANSWER HERE
        log_probability = 0.0
        
        log_probability += self.logprior[c]
        for word in doc.split():
            if word in self.vocabulary:
                if (word,c) in self.loglikelihood: #avoid key error to make sense when trying to access dictionary
                    log_probability += self.loglikelihood[(word,c)]

        return log_probability
        # >>> END YOUR ANSWER
                
    def predict(self, doc):
        """Return the most likely class for a given document under the trained classifier model.
        This should be only a few lines of code, and should make use of your self.score function.

        Consider using the `max` built-in function. There are a number of ways to do this:
           https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

        Parameters
        ----------
        doc : str
            A text representation of a document to score.
        
        Returns
        -------
        str
            The most likely class as predicted by the model.
        """
        # >>> YOUR ANSWER HERE
        scores = {}
        for c in self.classes:
            scores[c] = self.score(doc,c)
            
        return max(scores,key=scores.get)
        # >>> END YOUR ANSWER


    def evaluate(self, test_dir='haiti/test', target='relevant'):
        """Calculate a precision, recall, and F1 score for the model
        on a given test set.

        Note the 'target' parameter here, giving the name of the class
        to calculate relative to. So you can consider a True Positive
        to be an instance where the gold label for the document is the
        target and the model also predicts that label; a False Positive
        to be an instance where the gold label is *not* the target, but
        the model predicts that it is; and so on.

        Parameters
        ----------
        test_dir : str
            The path to a directory containing the test data.
        target : str
            The name of the class to calculate relative to. 

        Returns
        -------
        (float, float, float)
            The model's precision, recall, and F1 score relative to the
            target class.
        """        
        test_data = {c: os.path.join(test_dir, c) for c in self.classes}
        if not target in test_data:
            print('Error: target class does not exist in test data.')
            return
        outcomes = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        # >>> YOUR ANSWER HERE
        for c in test_data:
            c_test = open(test_data[c]).readlines()

        #true positives
            for doc in c_test:
                prediction = self.predict(doc)
                if prediction == c == 'relevant':
                    outcomes['TP'] += 1

        #true negatives
                elif prediction == c == 'irrelevant':
                    outcomes['TN'] += 1

        #false positives
                elif prediction == 'relevant' != c:
                    outcomes['FP'] += 1

        #false negatives
                elif prediction == 'irrelevant' != c:
                    outcomes['FN'] += 1

        precision = (outcomes['TP'])/(outcomes['TP']+outcomes['FP']) # replace with equation for precision
        recall = (outcomes['TP'])/(outcomes['TP']+outcomes['FN'])# replace with equation for recall
        f1_score = (2*precision*recall)/(precision+recall) # replace with equation for f1
        # >>> END YOUR ANSWER
        return (precision, recall, f1_score)


    def print_top_features(self, k=10):
        results = {c: {} for c in self.classes}
        for w in self.vocabulary:
            for c in self.classes:
                ratio = math.exp( self.loglikelihood[w, c] - min(self.loglikelihood[w, other_c] for other_c in self.classes if other_c != c) )
                results[c][w] = ratio

        for c in self.classes:
            print(f'Top features for class <{c.upper()}>')
            for w, ratio in sorted(results[c].items(), key = lambda x: x[1], reverse=True)[0:k]:
                print(f'\t{w}\t{ratio}')
            print('')
            
            
if __name__ == '__main__':
    target = 'relevant'

    clf = NaiveBayesClassifier(train_dir = 'haiti/train')
    clf.train()
    print(f'Performance on class <{target.upper()}>, keeping stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/dev', target = target)
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')
    
    clf = NaiveBayesClassifier(train_dir = 'haiti/train', REMOVE_STOPWORDS=True)
    clf.train()
    print(f'Performance on class <{target.upper()}>, removing stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/dev', target = target)
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')

    clf.print_top_features()


