import random, string, nltk
from collections import Counter
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize


# Makes sure the process is always the same and reproducible for everyone,
# so we can compare results and run the autograder.
random.seed(42) 
np.random.seed(42)

class LDATopicModel:

    def __init__(self, K=3, dataset='animals', iterations=100):
        # Hyperparameters
        self.num_topics = K
        self.topics = list(range(self.num_topics))
        self.iterations = iterations

        # symmetric, sparse dirichlet priors
        self.alpha = 1.0 / self.num_topics 
        self.beta = 1.0 / self.num_topics

        # Choose and load dataset
        if dataset == 'animals':
            self.documents = [[w.lower() for w in line.split()] for line in open('animals.txt')]
        elif dataset == '20_newsgroups_subset':
            self.documents = []
            for doc in fetch_20newsgroups(categories=('sci.space', 'misc.forsale', 'talk.politics.misc'), remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42, subset='train').data:
                tokenized = [w.lower() for w in word_tokenize(doc)]
                self.documents.append(tokenized)
        elif dataset == '20_newsgroups':
            self.documents = []
            for doc in fetch_20newsgroups(remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42, subset='train').data:
                tokenized = [w.lower() for w in word_tokenize(doc)]
                self.documents.append(tokenized)
        else:
            # As an extension, you could try loading in other datasets to run the model on.
            # You'll want to load in the documents as a list of lists, where each item in the
            # first list corresponds to the document, and each item within that corresponds
            # to a tokenized word.
            # >>> YOUR ANSWER HERE (optional)
            self.documents = []
            raise NotImplementedError # delete this if you add something here
            # >>> END YOUR ANSWER
        self.filter_docs()
        print('# Docs:', len(self.documents))


    def filter_docs(self):
        """
        Filtering the available features for LDA makes a big difference on the
        quality of the output. This function is given for you and filters out
        the words in the documents based on document frequency and other
        factors. Check it out if you're interested.
        """
        # Calculate document frequencies
        dfs = Counter()
        for doc in self.documents:
            for word in set(doc):
                dfs[word] += 1

        # Remove stopwords, punctuation, numbers, and too common / rare words
        to_remove = set([l.strip() for l in open('english.stop')] + ["n't", "'s"])        
        for word in dfs:
            df_proportion = dfs[word] / len(self.documents)
            if df_proportion < 0.005: # remove words appearing in less than 0.5% of documents
                to_remove.add(word) 
            elif df_proportion > 0.5: # remove words appearing in more than 50% of documents
                to_remove.add(word)
            elif all(c in string.punctuation + string.digits for c in word): # remove punctuation and numerical
                to_remove.add(word)
            elif len(word) < 3: # remove very short words
                to_remove.add(word)

        # Re-constitute the dataset
        filtered_documents = []
        for doc in self.documents:
            new_doc = []
            for word in doc:
                if not word in to_remove:
                    new_doc.append(word)
            if len(new_doc) > 0:
                filtered_documents.append(new_doc)
        self.documents = filtered_documents
        return
            

    def decrement_counts(self, doc_id, word, z):
        """
        Decrement (-= 1) global counts for this assignment of (doc_id, word, z).
        
        Parameters
        ----------
        doc_id : int
            The index (identifier) of the current document.

        word : str
            The word identity of the current word.

        z: int
            The number of the current topic.


        Returns
        -------
        None (updates self.doc_topic_counts, self.topic_word_counts,
              self.topic_counts, and self.doc_lengths)
        """
        # >>> YOUR ANSWER HERE
        
        self.doc_topic_counts[doc_id][z] -= 1
        self.topic_word_counts[z][word] -= 1
        self.topic_counts[z] -= 1
        self.doc_lengths[doc_id] -= 1

        # >>> END YOUR ANSWER
        return


    def increment_counts(self, doc_id, word, z):
        """
        Increment (+= 1) global counts for this assignment of (doc_id, word, z).
        
        Parameters
        ----------
        doc_id : int
            The index (identifier) of the current document.

        word : str
            The word identity of the current word.

        z: int
            The number of the current topic.

        Returns
        -------
        None (updates self.doc_topic_counts, self.topic_word_counts,
              self.topic_counts, and self.doc_lengths)
        """
        # >>> YOUR ANSWER HERE

        self.doc_topic_counts[doc_id][z] += 1
        self.topic_word_counts[z][word] += 1
        self.topic_counts[z] += 1
        self.doc_lengths[doc_id] += 1

        # >>> END YOUR ANSWER
        return


    def initialize_counts(self):
        """
        Initialize all the counts we want to keep track of through
        the training process.
        
        This function should loop through every word in every document
        and make a random assignment from the possible topics
        to every word - this is most straightforward by doing:
            random.choice(self.topics)

        Once you've made that assignment, you'll need to update all of our
        counters so we can use them to calculate quantities of interest later.        
        Therefore I strongly recommend saving the randomly generated topic
        assignment for each word in a variable called, e.g., `z`, which you
        can use as a key in each of these dictionaries.

        Specifically, we need to track:

        - self.vocabulary, set of strings
            represents all words used in the corpus

        - self.doc_topic_counts, nested dictionary of the form:
            self.doc_topic_counts[doc_id][z] = number of times
                words assigned to topic `z` appear in document `doc_id`
        
        - self.topic_word_counts, nested dictionary of the form:
            self.topic_word_counts[z][word] = number of times
                word `word` has been assigned to topic `z`

        - self.doc_lengths, dictionary of the form:
            self.doc_lengths[doc_id] = count of total words in
                document `doc_id`
        
        - self.topic_counts, dictionary of the form:
            self.topic_counts[z] = count of how many words total
                have been assigned to topic `z`

        - self.assignments, dictionary of the form:
            self.assignments[doc_id, word_id] = z, keeping track of
               which topic every word in every document has been
               assigned to.       

        Make sure you think through and understand what each object
        is counting and why.

        Also take careful note of the python type and conceptual meaning
        of each item we are using as a key. You can call these other things,
        but using the variable names I've been using above they are:

        - doc_id : int, index of a document in self.documents
        - word_id : int, index of a given word within a given document,
            so self.documents[doc_id][word_id] would return the string
            of this word.
        - word : str, word type and therefore a member of self.vocabulary
        - z : int, topic id and therefore a member of self.topics
       

        Parameters
        ----------
        None

        Returns
        -------
        None (updates self.doc_topic_counts, self.topic_word_counts,
              self.topic_counts, self.doc_lengths, self.vocabulary,
              and self.assignments)
        """
        # Collections of counts and assignments
        self.vocabulary = set([])
        self.doc_topic_counts = {doc_id: Counter() for doc_id in range(len(self.documents))}
        self.topic_word_counts = {z: Counter() for z in self.topics}
        self.doc_lengths = Counter()
        self.topic_counts = Counter()
        self.assignments = {}
        
        # Generate initial random topics and collect initial counts
        # >>> YOUR ANSWER HERE
        
        for doc_id in range(len(self.documents)):
            for word_id in range(len(self.documents[doc_id])):
                
                # Get word
                word = self.documents[doc_id][word_id]

                # For word, set z = random topic
                z = random.choice(self.topics)

                # Add word to self.vocab
                self.vocabulary.add(word)

                # Increment counts of
                    # self.doc_lengths[doc_id]
                    # self.doc_topic_counts[doc_id][z]
                    # self.topic_word_counts[z][word]
                    # self.topic_counts[z]
                self.increment_counts(doc_id, word, z)

                # Assign a topic to a word
                self.assignments[doc_id, word_id] = z

        # Print attributes
        # print("self.vocabulary = ", self.vocabulary)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("self.doc_topic_counts = ", self.doc_topic_counts)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("self.topic_word_counts = ", self.topic_word_counts)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("self.doc_lengths = ", self.doc_lengths)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("self.topic_counts = ", self.topic_counts)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("self.assignments = ", self.assignments)

        # >>> END YOUR ANSWER
        return


    def print_topics(self):
        """
        Given for you, this function prints out the words most associated with each topic
        using your self.phi_i_v function.
        """
        print('---------')
        for z in self.topics:
            vals = {}
            for word in self.vocabulary:
                vals[word] = self.phi_i_v(z, word)
            print('TOPIC',z,': ', ' '.join(str(w) + ' ' + '{:.3f}'.format(v) for w, v in sorted(vals.items(), key=lambda x: x[1], reverse=True)[0:10]))
        print('---------')            
        return


    def theta_d_i(self, doc_id, z):
        """
        Calculate the current document weight for this topic.

        This is given in Applications of Topic Models Ch 1., pg. 15, equation 1.2.

        You should use the Dirichlet parameter given in self.alpha in this calculation.

        Parameters
        ----------
        doc_id : int
            The index (identifier) of the current document.

        z: int
            The number of the current topic.

        Returns
        -------
        float
            The document weight for this topic (probability of this topic given this document).
        """
        # >>> YOUR ANSWER HERE
        
        N_d_i = self.doc_topic_counts[doc_id][z]
        a_i = self.alpha
        N_d_k = self.doc_lengths[doc_id]
        a_k = self.alpha*self.num_topics
        
        thet_d_i = (N_d_i + a_i)/(N_d_k+a_k)
        
        return thet_d_i
        # >>> END YOUR ANSWER


    def phi_i_v(self, z, word):
        """
        Calculate the current topic weight for this word.

        This is given in Applications of Topic Models Ch 1., pg. 15, equation 1.3.

        You should use the Dirichlet parameter given in self.beta in this calculation.

        Parameters
        ----------
        z: int
            The number of the current topic.

        word : str
            The word identity of the current word.

        Returns
        -------
        float
            The topic weight for this word (probability of this word given this topic).
        """
        # >>> YOUR ANSWER HERE

        V_i_v = self.topic_word_counts[z][word]
        B_v = self.beta
        V_i_w = self.topic_counts[z]
        B_w = self.beta*len(self.vocabulary)
        
        ph_i_v = (V_i_v + B_v)/(V_i_w+B_w)
        
        return ph_i_v
        # >>> END YOUR ANSWER

        
    def train(self):
        """
        Train the topic model using collapsed Gibbs sampling.

        Specifically, in each iteration of training, you should loop over
        each document in the corpus, and each word of each document.

        Then follow these steps:
        - Observe (and hold in a variable) the prior assignment of the
             current token.
        - Use your self.decrement_counts function to remove the current
             token from all our counts so it does not impact our 
             probability estimates.
        - Calculate the weight for each topic by multiplying together
             the results of self.theta_d_i and self.phi_i_v for the current
             document and word. This is equation 1.4 in Applications of 
             Topic Models, on pg. 16 in Ch. 1.
        - Use these weights to randomly sample a new topic for this token.
             Your probability of choosing the topic should be proportional
             to the weight calculated in the previous step.
             For consistency with the autograder I suggest using the 
             built-in `random` library's `random.choices` function,
             which takes a `weights` argument. Another option is numpy's
             `np.random.choice` function, but it's a bit more complicated
             so I don't particularly recommend it for this application. 
        - Update `self.assignments` with this new assignment, and use
             your self.increment_counts function to add this token back
             in to the counts with its new assignment.


        Parameters
        ----------
        None

        Returns
        -------
        None (updates self.assignments and other count dictionaries)
        """
        self.initialize_counts()

        for iteration in range(self.iterations):
            if iteration % 10 == 0:
                print("\n\nIteration:", iteration)
                self.print_topics()
            else:
                print(iteration, end=' ', flush=True)

            # >>> YOUR ANSWER HERE
            
            for doc_id in range(len(self.documents)):
                for word_id in range(len(self.documents[doc_id])):

                    # Get word
                    word = self.documents[doc_id][word_id]

                    # Get previous topic assignment for word
                    prev_ass = self.assignments[doc_id, word_id]

                    # Decrement
                    self.decrement_counts(doc_id, word, prev_ass)

                    # Calculate weight for each topic
                    topic_weights = {}
                    for topic in self.topics:
                        weight = self.theta_d_i(doc_id, topic)*self.phi_i_v(topic, word)
                        topic_weights[topic] = weight

                    # Assign a new topic for word based on weights
                    new_ass  = random.choices(list(topic_weights.keys()), weights=list(topic_weights.values()), k=1)[0]

                    # Update word assignment
                    self.assignments[doc_id, word_id] = new_ass

                    # Incremement
                    self.increment_counts(doc_id, word, new_ass)

            # >>> END YOUR ANSWER

        print('\n\nTraining Complete')
        self.print_topics()



if __name__ == '__main__':
    """
    Feel free to modify the below to play around with this as you are interested. 
    You can change 'K' to modify the number of learned topics, or 'dataset' to change which dataset to use.

    Available datasets already coded in include:
      - 'animals'                 a toy dataset of documents about three kinds of animals
      - '20_newsgroups_subset'    a subset of three categories of the classic '20 newsgroups' dataset
      - '20_newsgroups'           the entire 20 newsgroups; note this will take a relatively long time to run.
                                  this will also do better with more topics, since indeed it has more than 3.
                                  for me to run this dataset for 100 iterations with 20 topics takes about 40 minutes.
    """

    lda = LDATopicModel(K=3, dataset='animals')
    lda.train()

                


