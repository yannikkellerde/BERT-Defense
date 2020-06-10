import numpy as np
import util
from tqdm import tqdm,trange
from operator import itemgetter


def levenshteinDistance(target, source, word_embedding):
    char_app = char_apparence(source)
    vowls_in = vowl_checker(source)
    n = len(target)
    m = len(source)
    distance = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        distance[i][0] = i
    for j in range(1, m+1):
        distance[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if target[i-1] == source[j-1]:
                distance[i][j] = distance[i-1][j-1]
            else:
                distance[i][j] = min((distance[i-1][j] + in_cost(target[i-1], vowls_in),# insertion von target_i in source
                                      distance[i-1][j-1] + sub_cost(target[i-1], source[j-1], word_embedding),# substituition von target in source
                                      distance[i][j-1] + del_cost(source[j-1], char_app)))# delition  source_j
    return distance[n][m]


def in_cost(in_char, vowls_in):
    vowls = ["A", "a", "E", "e", "I", "i", "O", "o", "u", "U" ]
    if (in_char in vowls) and not(vowls_in):
        return 0.2
    else:
        return 1


def sub_cost(char1, char2, word_embedding):
    vek1 = word_embedding[ord(char1)]
    vek2 = word_embedding[ord(char2)]
    return 1 - (vek1@vek2)/2


def del_cost(del_char, table, scaler=0.75):
    return scaler**(table[del_char]-1)


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
            table[c] = 1
    return table


def get_word_dic_distance(word, dic, word_embedding):
    distance = []
    for sample_word in tqdm(dic):
        distance.append((sample_word, levenshteinDistance(sample_word, word, word_embedding)))
    distance.sort(key=itemgetter(1))
    words, values = zip(*distance)
    values = np.array(values)
    max_value = np.max(values)
    values = max_value - values
    values = util.softmax(np.array(values),theta=4)
    distance = list(zip(words, values))
    return distance


if __name__ == '__main__':
    print(vowl_checker("Hallo"))
    dic = util.load_dictionary("DATA/wiki-100k.txt")
    word_embedding = util.load_pickle("visual_embeddings.pkl")
    distance = get_word_dic_distance("Бahrзiň", dic, word_embedding)
    for i in range(10):
        print(distance[i])
