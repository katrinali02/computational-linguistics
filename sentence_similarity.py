import csv, string
import numpy as np
from scipy.stats import spearmanr
from embeddings import Embeddings
from nltk import pos_tag
from nltk.tokenize import word_tokenize



def read_sts(infile = 'data/sts-dev.csv'):
    sts = {}
    for row in csv.reader(open(infile), delimiter='\t'):
        if len(row) < 7: continue
        val = float(row[4])
        s1, s2 = row[5], row[6]
        sts[s1, s2] = val / 5.0
    return sts

def calculate_sentence_embedding(embeddings, sent, weighted = False):
    """
    Calculate a sentence embedding vector.

    If weighted is False, this is the elementwise sum of the constituent word vectors.
    If weighted is True, multiply each vector by a scalar calculated
    by taking the log of its word_rank. The word_rank value is available
    via a dictionary on the Embeddings class, e.g.:
       embeddings.word_rank['the'] # returns 1

    In either case, tokenize the sentence with the `word_tokenize` function,
    lowercase the tokens, and ignore any words for which we don't have word vectors. 

    Parameters
    ----------
    sent : str
        A sentence for which to calculate an embedding.

    weighted : bool
        Whether or not to use word_rank weighting.

    Returns
    -------
    np.array of floats
        Embedding vector for the sentence.
    
    """
    # >>> YOUR ANSWER HERE

    sent_embedding = 0.0

    # Tokenize sent
    sent_tokenized = [x.lower() for x in word_tokenize(sent)]
    for word in sent_tokenized:
        if word not in embeddings:
            sent_tokenized.pop(word)

    # Simple sum
    if weighted == False:
        for word in sent_tokenized:
            sent_embedding += embeddings[word]
    
    # Weighted sum
    if weighted == True:
        for word in sent_tokenized:
            #np.append(sent_embedding, embeddings[word]*np.log(embeddings.word_rank[word]))
            sent_embedding += embeddings[word]*np.log(embeddings.word_rank[word])

    return sent_embedding
    
    # >>> END YOUR ANSWER



def score_sentence_dataset(embeddings, dataset, weighted = False):
    """
    Calculate the correlation between human judgments of sentence similarity
    and the scores given by using sentence embeddings.

    Parameters
    ----------
    dataset : dictionary of the form { (sentence, sentence) : similarity_value }
        Dataset of sentence pairs and human similarity judgments.
    
    weighted : bool
        Whether or not to use word_rank weighting.

    Returns
    -------
    float
        The Spearman's Rho ranked correlation coefficient between
        the sentence emedding similarities and the human judgments.     
    """
    # >>> YOUR ANSWER HERE

    gold_sim_vals = []
    model_vals  = []

    for sent_pair in list(dataset):
        
        # Gold
        gold_val = dataset[sent_pair]
        gold_sim_vals.append(gold_val)
        
        # Model
        sent1_embedding = calculate_sentence_embedding(embeddings,sent_pair[0], weighted)
        sent2_embedding = calculate_sentence_embedding(embeddings,sent_pair[1],weighted)
        model_val = embeddings.cosine_similarity(sent1_embedding, sent2_embedding)
        model_vals.append(model_val)

    spearman_r = spearmanr(gold_sim_vals, model_vals)[0]
    
    return spearman_r

    # >>> END YOUR ANSWER

if __name__ == '__main__':
    embeddings = Embeddings()
    sts = read_sts()
    
    print('STS-B score without weighting:', score_sentence_dataset(embeddings, sts))
    print('STS-B score with weighting:', score_sentence_dataset(embeddings, sts, True))
