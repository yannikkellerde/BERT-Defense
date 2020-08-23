import sys
sys.path.append("..")
import numpy as np
import util
from tqdm import tqdm,trange
from edit_distance.edit_distance import get_word_dic_distance
# https://github.com/yannikbenz/robustness-of-deep-learning-models-to-low-level-adversarial-attacks/blob/master/code/attacks/simple_attacks.py
def swap(word: str, inner: bool, seed=None):
    """Shuffles the chars in each word. If inner is set the first and last letters position remain untouched.
    >>> swap("hello world", True, 56)
    hlelo wlord
    >>> swap("hello word", False, 42)
    eolhl odrwl
    :param word:
    :param seed: seed
    :param inner: if set, only the inner part of the word will be swapped
    :return: swapped text
    """

    def __shuffle_string__(_word: str, _seed=seed):
        """
        shuffles the given string if a seed is given it shuffles in respect to the given seed.
        hello world -> elloh roldw
        :param _seed: seed
        :param _word: string (word) to shuffle
        :return: shuffled string
        """
        chars = list(_word)
        if _seed is not None:
            np.random.seed(_seed)
        np.random.shuffle(chars)
        return ''.join(chars)

    if len(word) < 3 or inner and len(word) < 4:
        return word
    perturbed = word
    tries = 0
    while perturbed == word:
        tries += 1  # we can get a deadlock if the word is e.g. maas
        if tries > 10:
            break
        if inner:
            first, mid, last = word[0], word[1:-1], word[-1]
            perturbed = first + __shuffle_string__(mid) + last
        else:
            perturbed = __shuffle_string__(word)
    return perturbed

def append_each_word_suffled_to_list(sentence, suffled_words, inner=False):
    new_words = sentence.split()
    for word in new_words:
        suffled_words.append((word, swap(word, inner=inner)))
    return suffled_words


def test_prior(cheap_actions, inner):
    letter_begin = util.load_dictionary("../../DATA/dictionaries/bert_letter_begin.txt")
    number_begin = util.load_dictionary("../../DATA/dictionaries/bert_number_begin.txt")
    dic = letter_begin+number_begin
    word_embedding = util.load_pickle("../binaries/visual_embeddings.pkl")
    with open("../../phonetic_attacks/sts-b-sentences.txt","r") as f:
            sentences = f.read().splitlines()
    suffled_words = [] 
    for sentence in sentences:
        sentence = sentence.lower()
        suffled_words = append_each_word_suffled_to_list(sentence, suffled_words, inner=inner)
    index_list = []
    prob_list = [] 
    for word_pair in tqdm(suffled_words):
        distance = get_word_dic_distance(word_pair[1], dic, word_embedding, cheap_actions=cheap_actions, progress=False, cheap_deletions=False)
        element = list(filter(lambda x: x[0]==word_pair[0], distance))
        if element:
            index = distance.index(element[0])
            index_list.append(index+1)
            prob_list.append(element[0][1])
    print("Mean Position: ", np.mean(np.array(index_list)))
    print("Mean Probability: ", np.mean(np.array(prob_list)))

if __name__ == '__main__':
    print("--------------------------------------------------------------")
    print("Test Prior: cheap_actions=True, inner=True")
    test_prior(True, True)
    print("--------------------------------------------------------------")
    print("Test Prior: cheap_actions=True, inner=False")
    test_prior(True, False)
    print("--------------------------------------------------------------")
    print("Test Prior: cheap_actions=False, inner=True")
    test_prior(False, True)
    print("--------------------------------------------------------------")
    print("Test Prior: cheap_actions=False, inner=False")
    test_prior(False, False)
    print("--------------------------------------------------------------")
    
        
