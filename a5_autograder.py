import sys, os, math
import numpy as np
sys.path.append(os.getcwd())
from embeddings import Embeddings

errs = 0
embeddings = Embeddings('glove_top50k_50d.txt')

print("Double-checking the embeddings are reading properly...")
try:
    assert embeddings['banana'][32] == -0.79569
    print('\tlooks good!')
except AssertionError:
    print('\thmm, somehow your embeddings are not matching up. Did you modify the "__init__" code?')
    errs += 1

print("Checking cosine similarity...")
val = embeddings.cosine_similarity('cat','hat')
val2 = embeddings.cosine_similarity('hat','cat')
sim_err = False
try:
    assert val == val2
except AssertionError:
    print(f"\terror, similarity should be symmetrical, but got {val} for sim('cat','hat') and {val2} for sim('hat','cat')")
    sim_err = True
    errs += 1
try:
    assert math.fabs(val - 0.6289424581701398) < 0.001
except AssertionError:
    print(f"\terror, expected similarity of ~0.6289 for sim('cat','hat') but got {val}")
    sim_err = True
    errs += 1
if not sim_err:
    print('\tlooks good!')


print("Checking most similar...")
try:
    top = embeddings.most_similar('goat', exclude=['goat'])[0]
    assert top[0] == 'potato'
    print('\tlooks good!')
except AssertionError:
    print(f"\terror, expected most similar to 'goat' to be 'potato' but got '{top[0]}'")
    errs += 1


test_dataset = {('boat', 'float'): 0.5, ('moat','coat'): 0.1, ('note','vote'): 0.9}
from word_similarity import score_word_dataset
print("Checking word similarity scoring...")
try:
    val = score_word_dataset(embeddings, test_dataset)
    assert val == 0.5
    print('\tlooks good!')
except AssertionError:
    print(f"\terror, seems to be a problem here - not working on a toy dataset.")
    errs += 1

test_dataset = {('this one', 'that one'): 0.6, ('all of them', 'none of them'): 0.2, ('many people say', 'sources indicate'): 0.8}

from sentence_similarity import calculate_sentence_embedding, score_sentence_dataset
print('Checking sentence embedding calculation...')
try:
    val = np.mean(calculate_sentence_embedding(embeddings, 'Over, the rainbow?', weighted=False))
    val2 = np.mean(calculate_sentence_embedding(embeddings, 'Over, the rainbow?', weighted=True))
    assert math.fabs(val + 0.147621) < 0.0001
    assert math.fabs(val2 + 0.960183) < 0.0001
    print('\tlooks good!')
except AssertionError:
    print(f"\terror, sentence embeddings look off - taking the mean of the embedding for 'Over, the rainbow?' expected about -0.1476 for unweighted and -0.9602 for weighted, but got {val} and {val2} respectively")
    errs += 1

print('Checking sentence similarity scoring...')
try:
    val = score_sentence_dataset(embeddings, test_dataset, weighted=True)
    assert val == -0.5
    print('\tlooks good!')
except AssertionError:
    print(f"\terror, seems to be a problem here - not working on a toy dataset.")
    errs += 1


from analogies import read_turney_analogies, answer_by_analogy, answer_by_parallelism, evaluate
SAT_questions = read_turney_analogies(embeddings, path='data/SAT-package-V3.txt')
test_q = SAT_questions[40]
print('Double-checking analogies are reading properly...')
try:
    assert test_q['question'] == ('calculator','compute')
    print('\tlooks good!')
except AssertionError:
    print(f"\terror, did you modify the read_turney_analogies function, or the SAT-package-V3.txt file?")
    errs += 1


print('Checking analogy answering methods...')
try:
    a_answer = answer_by_analogy(embeddings, test_q['question'], test_q['choices'])
    p_answer = answer_by_parallelism(embeddings, test_q['question'], test_q['choices'])
    assert a_answer == 0
    assert p_answer == 2
    print('\tlooks good!')
except AssertionError:
    print(f'\terror, for question 40 expected answer_by_analogy of 0 and answer_by_parallelism of 2, got {a_answer} and {p_answer} respectively')
    errs += 1

q_subset = SAT_questions[20:50]
print('Checking evaluation...')
try:
    analogy_result = evaluate(embeddings, q_subset, answer_by_analogy)
    parallelism_result = evaluate(embeddings, q_subset, answer_by_parallelism)
    assert math.fabs(0.233333 - analogy_result) < 0.001
    assert math.fabs(0.266666 - parallelism_result) < 0.001
    print('\tlooks good!')
except AssertionError:
    print(f'\terror, on a subset of questions expected result by analogy of ~0.2333 and by parallelism of ~0.2666, got {analogy_result} and {parallelism_result} respectively')
    errs += 1

if errs == 0:
    print('All tests passed!\n')
else:
    print("Seems like there's still some work to do.")

