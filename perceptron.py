import os, math, random
from collections import Counter, defaultdict

random.seed(42)

class AveragedPerceptronClassifier:
    """Code for a Averaged Perceptron Classifier making a binary classification.
    """
    
    def __init__(self, train_dir='haiti/train', REMOVE_STOPWORDS=False):
        self.REMOVE_STOPWORDS = REMOVE_STOPWORDS
        self.stopwords = set([l.strip() for l in open('english.stop')])
        self.classes = os.listdir(train_dir)
        self.vocabulary = set([])
        self.weights = defaultdict(float) # mapping of features (words) to weights
        self.train_data = {c: os.path.join(train_dir, c) for c in self.classes} # mapping of class to file path containing data for that class
        self.num_iters = 10000 # arbitrary

        
    def train(self):
        """Train the Averaged Perceptron classification model, roughly following the pseudocode for
        training given in Algorithm 4 of NLP Notes Chapter 2.

        Specifically, you want to start from your weights being all set to 0. Collect your vocabulary
        and remove stopwords if necessary. Then self.num_iters times, randomly choose a document from
        the training data, and calculate its most likely class given the current weights. If this guess
        is correct, continue to the next iteration. If it is incorrect, call self.update_weights to 
        nudge the weights in the direction of the correct answer.

        To make this an averaged perceptron, as you go accumulate a running sum for each weight at each
        iteration. At the end of this process set each final weight to its total sum divided by its running
        sum.

        Parameters
        ----------
        None (reads training data from self.train_data, and the number of iterations from self.num_iters)
        
        Returns
        -------
        None (updates class attribute self.weights)
        """
        # >>> YOUR ANSWER HERE
        pass
        # >>> END YOUR ANSWER

        
                
    def score(self, doc):
        """Score a given document with the current set of weights.

        This is tantamount to summing the weights for the words in the document.
        If the score (sum) is higher than 0, return self.classes[0]; otherwise
        return self.classes[1].

        Parameters
        -------
        doc : str
           The document to score.

        Returns
        -------
        str
           The name of the predicted class.        
        """

        # >>> YOUR ANSWER HERE
        return self.classes[0]
        # >>> END YOUR ANSWER


    def update_weights(self, doc, c):
        """Perform a weight update, knowing that the given document was
        predicted incorrectly with the current weights for gold class c.

        You should iterate over the words in the document, and update the weights
        for each one by +1.0 or -1.0, depending on which class we want to be
        better at next time. Recall from the score function above that 
        we're saying self.classes[0] should be associated with positive weights.      

        Parameters
        ----------
        doc : str
            The document which we predicted incorrectly.
        c : str
            The class we should have predicted.

        """
        # >>> YOUR ANSWER HERE
        pass
        # >>> END YOUR ANSWER

        
    def evaluate(self, test_dir='haiti/test', target='relevant'):
        """Calculate a precision, recall, and F1 score for the model
        on a given test set.

        Not the 'target' parameter here, giving the name of the class
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

        precision = 0.0 # replace with equation for precision
        recall = 0.0 # replace with equation for recall
        f1_score = 0.0 # replace with equation for f1
        # >>> END YOUR ANSWER
        return (precision, recall, f1_score)


    def print_top_features(self, k=10):
        print(f'Top features for class <{self.classes[0].upper()}>')
        for word, weight in sorted(self.weights.items(), key = lambda x: x[1], reverse=True)[0:k]:
            print(f'\t{word}\t{weight}')
        print('')

        print(f'Top features for class <{self.classes[1].upper()}>')
        for word, weight in sorted(self.weights.items(), key = lambda x: x[1], reverse=False)[0:k]:
            print(f'\t{word}\t{weight}')
        print('')
            
            
if __name__ == '__main__':
    clf = AveragedPerceptronClassifier(train_dir = 'haiti/train')
    clf.train()
    target = 'relevant'
    print(f'Performance on class <{target.upper()}>, keeping stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/dev', target = target)
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')
    
    clf = AveragedPerceptronClassifier(train_dir = 'haiti/train', REMOVE_STOPWORDS=True)
    clf.train()
    print(f'Performance on class <{target.upper()}>, removing stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/dev', target = target)
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')
    clf.evaluate(test_dir = 'haiti/test')

    clf.print_top_features()


