from numpy.core.fromnumeric import argmax
from embeddings import Embeddings


def read_turney_analogies(embeddings, path = 'data/SAT-package-V3.txt'):    
    questions = []
    letters = ['a','b','c','d','e']

    current = {'choices': []}
    start = True
    for line in open(path):
        if line.startswith('190 FROM REAL SATs') or line.strip() == '' or line.startswith('KS') or line.startswith('ML'): continue
        if len(line.strip()) == 1:
            current['answer'] = letters.index(line.strip())
            all_words = []
            all_words.extend(current['question'])
            for item in current['choices']:
                all_words.extend(item)
            if all(w in embeddings for w in all_words): 
                questions.append(current)
            current = {'choices':[]}
            start = True
            continue
        if start:
            current['question'] = tuple(line.split()[0:2])
            start = False
        else:
            current['choices'].append(tuple(line.split()[0:2]))
    return questions


def answer_by_analogy(embeddings, question, choices):
    """
    Answer an analogy question by the analogy (parallelogram) method.

    For a question a:b and possible choices of the form aa:bb,
    the answer is the one that maximizes cos(a - b + bb, aa).

    Parameters
    ----------
    question : tuple of (word, word)
       Words a and b to target.

    choices : list of tuples of (word, word)
       List of possible analogy matches aa and bb.

    Returns
    -------
    int
       index into `choices` of the estimated answer.
    """
    # >>> YOUR ANSWER HERE
    
    correct_choice_index = 0
    choice_cos = []
    
    a_embed = embeddings.__getitem__(question[0])
    b_embed = embeddings.__getitem__(question[1])

    for choice in choices:
        aa_embed = embeddings.__getitem__(choice[0])
        bb_embed = embeddings.__getitem__(choice[1])
        cos = embeddings.cosine_similarity((a_embed-b_embed+bb_embed), aa_embed)
        choice_cos.append(cos)

    correct_choice_index = argmax(choice_cos)

    return correct_choice_index
    # >>> END YOUR ANSWER

def answer_by_parallelism(embeddings, question, choices):
    """
    Answer an analogy question by a parallelism method.

    For a question a:b and possible choices of the form aa:bb,
    the answer is the one that maximizes cos(a - b, aa - bb).

    Parameters
    ----------
    question : tuple of (word, word)
       Words a and b to target.

    choices : list of tuples of (word, word)
       List of possible analogy matches aa and bb.

    Returns
    -------
    int
       index into `choices` of the estimated answer.
    """
    # >>> YOUR ANSWER HERE

    correct_choice_index = 0
    choice_cos = []
    
    a_embed = embeddings.__getitem__(question[0])
    b_embed = embeddings.__getitem__(question[1])

    for choice in choices:
        aa_embed = embeddings.__getitem__(choice[0])
        bb_embed = embeddings.__getitem__(choice[1])
        cos = embeddings.cosine_similarity((a_embed-b_embed), (aa_embed-bb_embed))
        choice_cos.append(cos)

    correct_choice_index = argmax(choice_cos)

    return correct_choice_index

    # >>> END YOUR ANSWER

def evaluate(embeddings, dataset, method = answer_by_analogy):
    """
    Evaluate the guesses made by a given method.

    Parameters
    ----------
    dataset : list of dicts of the form {'question': (a, b), 'choices': [(aa, bb), ...], 'answer': idx}
        Represents a list of SAT analogy questions.

    method : func (either answer_by_analogy or answer_by_parallelism)
        The method to use. Note that in python you can pass functions
        along in this way without calling them, so inside this function
        you can call whichever method gets passed by doing `method(args)`.

    Returns
    -------
    float
        The accuracy of the given method: num_correct / num_total.
    """
    # >>> YOUR ANSWER HERE
    
    num_correct = 0
    num_total = 0

    for item in dataset:
        q = item['question']
        c = item['choices']
        guess = method(embeddings, q, c)
        if guess == item['answer']:
            num_correct += 1
        num_total += 1

    return num_correct / num_total

    # >>> END YOUR ANSWER


if __name__ == '__main__':
    embeddings = Embeddings()
    SAT_questions = read_turney_analogies(embeddings)
    
    analogy_result = evaluate(embeddings, SAT_questions, answer_by_analogy)
    parallelism_result = evaluate(embeddings, SAT_questions, answer_by_parallelism)
    print('Answering by analogy scored:',analogy_result)
    print('Answering by parallelism scored:',parallelism_result)
