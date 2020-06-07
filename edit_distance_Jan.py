import numpy as np
import util

def levenshteinDistance(target, source, word_embedding):
    n = len(target)
    m = len(source)
    distance = np.zeros((n+1, m+1))
    for i in range(1,n+1):
        distance[i][0] = i
    for j in range(1,m+1):
        distance[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if target[i-1]==source[j-1]:
                distance[i][j] = distance[i-1][j-1]
            else:
                distance[i][j] = min((distance[i-1][j] + in_cost(target[i-1]),       # insertion von target_i in source
                                      distance[i-1][j-1] + sub_cost(target[i-1], source[j-1], word_embedding),     # substituition von target in source
                                      distance[i][j-1] + del_cost(source[j-1])))      # delition  source_j
    return distance[n][m]

def in_cost(in_char):
    return 1

def sub_cost(char1, char2, word_embedding):
    vek1 = word_embedding[ord(char1)]
    vek2 = word_embedding[ord(char2)]
    return 1 - util.cosine_similarity(vek1, vek2)

def del_cost(del_char):
    return 1



if __name__ == '__main__':
    dic = util.load_dictionary("DATA/wiki-100k.txt")
    word_embedding = util.load_pickle("visual_embeddings.pkl")
    print(util.cosine_similarity(word_embedding[ord("W")], word_embedding[ord("T")]))
    print(levenshteinDistance("HallW","Hallo", word_embedding))
