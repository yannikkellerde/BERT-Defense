import sys
sys.path.append("..")
import numpy as np
from util.utility import get_full_word_dict,fast_argmin,softmax,load_pickle
from tqdm import tqdm,trange
from operator import itemgetter
from util.letter_stuff import trenner_punctuations,vocals, annoying_boys

"""
cheap_actions = an array d with a true or false statment for each action
                          d[0] == True: activate cheap insertion
                          d[1] == True: activate cheap substitution (need word_embedding)
                          d[2] == True: activate cheap deletion
                          d[3] == True: activate cheap transposition
                          shortcut: True activates all and False deactivated all
"""
def levenshteinDistance(target, source, cheap_actions=False , word_embedding=None, char_app=None,vowls_in=None):
    if not(cheap_actions is list):
        if cheap_actions:
            cheap_actions = np.ones(4, dtype=bool)
        else:
            cheap_actions = np.zeros(4, dtype=bool)
    if cheap_actions[1] and word_embedding is None:
        raise AttributeError("need word_embedding")
    if cheap_actions[3]:
        anagramm = anagramm_finder(source, target)
    if char_app is None and cheap_actions[2]:
        char_app = char_apparence(source)
    if vowls_in is None and cheap_actions[0]:
        vowls_in = vowl_checker(source)
    n = len(target)
    m = len(source)
    distance = np.zeros((n+1, m+1))
    i = 0
    for num in range(1, n+1):
        i += in_cost(target[num-1],vowls_in) if cheap_actions[0] else 1
        distance[num][0] = i
    j = 0
    for num in range(1, m+1):
        j += del_cost(source[num-1],char_app) if cheap_actions[2] else 1
        distance[0][num] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            possibilities = [distance[i-1][j] + (in_cost(target[i-1], vowls_in) if cheap_actions[0] else 1),# insertion von target_i in source
                                distance[i-1][j-1] + (sub_cost(target[i-1], source[j-1], word_embedding) if cheap_actions[1] else 1),# substituition von target in source
                                distance[i][j-1] + (del_cost(source[j-1], char_app) if cheap_actions[2] else 1)]# deletion  source_j
            if target[i-1] == source[j-1]:
                possibilities.append(distance[i-1][j-1])
            if i > 2 and j > 2 and target[i-1] == source[j-2] and target[i-2] == source[j-1]:
                possibilities.append(distance[i-2][j-2] + (trans_cost(anagramm) if cheap_actions[3] else 1))
            distance[i][j] = min(possibilities)
    return distance[n][m]


def in_cost(in_char, vowls_in):
    if (in_char in vocals) and not(vowls_in):
        return 0.3
    else:
        return 1


def sub_cost(char1, char2, word_embedding):
    vek1 = word_embedding[ord(char1)]
    vek2 = word_embedding[ord(char2)]
    return min((1 - vek1@vek2)*2,1)


def del_cost(del_char, table, scaler=0.75):
    scal_cost = scaler**(table[del_char]-1)
    return scal_cost


def trans_cost(anagramm):
    if anagramm:
        return 0.3
    else:
        return 1


def anagramm_finder(source, target):
    if not(len(source) == len(target)):
        return False
    for char in source:
        if not(char in target):
            return False
    return True


def vowl_checker(word):
    vowls = ["A", "a", "E", "e", "I", "i", "O", "o", "u", "U" ]
    for char in word:
        if char in vowls:
            return True
    return False


def char_apparence(word):
    table = {}
    for c in word:
        if c in table:
            table[c] = table[c] + 1
        else:
            if c in annoying_boys:
                table[c] = 3
            else:
                table[c] = 1
    return table

default_word_embedding = load_pickle("../binaries/visual_embeddings.pkl")
def get_word_dic_distance(word, dic, word_embedding=None, cheap_actions=False, keep_order=False, progress=True,theta=27):
    if word_embedding is None:
        word_embedding = default_word_embedding
    char_app = char_apparence(word)
    vowls_in = vowl_checker(word)
    if keep_order:
        distance = np.empty(len(dic))
        for i,sample_word in (tqdm(enumerate(dic)) if progress else enumerate(dic)):
            distance[i] = levenshteinDistance(sample_word, word,cheap_actions=cheap_actions,
                                                            word_embedding=word_embedding,
                                                            char_app=char_app,vowls_in=vowls_in)
        distance = softmax(1/(distance+1),theta=theta)
    else:
        distance = []
        for sample_word in (tqdm(dic) if progress else dic):
            distance.append((sample_word, levenshteinDistance(sample_word, word,cheap_actions=cheap_actions,
                                                                word_embedding=word_embedding,
                                                                char_app=char_app,vowls_in=vowls_in)))
        distance.sort(key=itemgetter(1))
        words, values = zip(*distance)
        values = softmax(1/(np.array(values)+1),theta=theta)
        distance = list(map(list,zip(words, values)))
    return distance


if __name__ == '__main__':
    dic = get_full_word_dict()
    #print(levenshteinDistance("eco", "eco" , cheap_actions=True, word_embedding=default_word_embedding))
    distance = get_word_dic_distance("spèedіnΜ", dic, cheap_actions=True)
    for w,p in distance:
        if w=="a" or w=="speeding":
            print(w,p)
    for i in range(40):
        print(distance[i])