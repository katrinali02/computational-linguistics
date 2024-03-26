import math
import numpy as np

class Embeddings:

    def __init__(self, glove_file = 'glove_top50k_50d.txt'): #quest path: '/projects/e31408/data/a5/glove_top50k_50d.txt'
        self.embeddings = {}
        self.word_rank = {}
        for idx, line in enumerate(open(glove_file)):
            row = line.split()
            word = row[0]
            vals = np.array([float(x) for x in row[1:]])
            self.embeddings[word] = vals
            self.word_rank[word] = idx + 1

    def __getitem__(self, word):
        return self.embeddings[word]

    def __contains__(self, word):
        return word in self.embeddings

    def vector_norm(self, vec):
        """
        Calculate the vector norm (aka length) of a vector.

        This is given in SLP Ch. 6, equation 6.8. For more information:
        https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

        Parameters
        ----------
        vec : np.array
            An embedding vector.

        Returns
        -------
        float
            The length (L2 norm, Euclidean norm) of the input vector.
        """
        # >>> YOUR ANSWER HERE

        vector_length = np.linalg.norm(vec)
        return vector_length 

        # >>> END YOUR ANSWER

    def cosine_similarity(self, v1, v2):
        """
        Calculate cosine similarity between v1 and v2; these could be
        either words or numpy vectors.

        If either or both are words (e.g., type(v#) == str), replace them 
        with their corresponding numpy vectors before calculating similarity.

        Parameters
        ----------
        v1, v2 : str or np.array
            The words or vectors for which to calculate similarity.

        Returns
        -------
        float
            The cosine similarity between v1 and v2.
        """
        # >>> YOUR ANSWER HERE
        if type(v1) == str:
            v1 = self.embeddings[v1]
        if type(v2) == str:
            v2 = self.embeddings[v2]

        numer = np.dot(v1, v2)
        denom = self.vector_norm(v1)*self.vector_norm(v2)

        return numer/denom

        # >>> END YOUR ANSWER

    def most_similar(self, vec, n = 5, exclude = []):
        """
        Return the most similar words to `vec` and their similarities. 
        As in the cosine similarity function, allow words or embeddings as input.


        Parameters
        ----------
        vec : str or np.array
            Input to calculate similarity against.

        n : int
            Number of results to return. Defaults to 5.

        exclude : list of str
            Do not include any words in this list in what you return.

        Returns
        -------
        list of ('word', similarity_score) tuples
            The top n results.        
        """
        # >>> YOUR ANSWER HERE

        most_sim = []
        for word in list(self.embeddings):
            if (word != vec) & (word not in exclude):
                sim = self.cosine_similarity(word, vec)
                word_sim_pair = (word, sim)
                most_sim.append(word_sim_pair)

        sorted_most_sim = sorted(most_sim, key=lambda tup: tup[1], reverse=True)

        final_most_sim = []
        for count, word_sim_pair in enumerate(sorted_most_sim):
            final_most_sim.append(word_sim_pair)
            if count == n:
                break

        return final_most_sim

        # >>> END YOUR ANSWER


if __name__ == '__main__':
    embeddings = Embeddings()
    word = 'man'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])

    word = 'woman'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])

    word = 'caucasian'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])

    word = 'african-american'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])
    
    word = 'american'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])

    word = 'chinese'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])

    word = 'christian'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])

    word = 'muslim'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])

    word = 'old'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])

    word = 'young'
    print(f'Most similar to {word}:')
    for item in embeddings.most_similar(word, exclude=[word]):
        print('\t',item[0], '\t', item[1])

