import numpy as np
import util
from tqdm import tqdm,trange
from operator import itemgetter
from letter_stuff import trenner_punctuations,vocals, annoying_boys


def levenshteinDistance(target, source, word_embedding,char_app=None,vowls_in=None,cheap_deletions=True):
    if char_app is None:
        char_app = char_apparence(source)
    if vowls_in is None:
        vowls_in = vowl_checker(source)
    n = len(target)
    m = len(source)
    freelo_amount = (0.2 if source[0] in trenner_punctuations else 0.5) if len(source)>2 else 1
    dels = 0
    distance = np.zeros((n+1, m+1))
    i = 0
    for num in range(1, n+1):
        i += in_cost(target[num-1],vowls_in)
        distance[num][0] = i
    j = 0
    for num in range(1, m+1):
        j += del_cost(source[num-1],char_app,False,freelo_amount)
        distance[0][num] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            possibilities = [distance[i-1][j] + in_cost(target[i-1], vowls_in),# insertion von target_i in source
                                distance[i-1][j-1] + sub_cost(target[i-1], source[j-1], word_embedding),# substituition von target in source
                                distance[i][j-1] + del_cost(source[j-1], char_app,i==n and cheap_deletions,freelo_amount)]# delition  source_j
            if target[i-1] == source[j-1]:
                possibilities.append(distance[i-1][j-1])
            choice = util.fast_argmin(possibilities)
            if i==n:
                if choice==2:
                    dels+=1
                else:
                    dels=0
            distance[i][j] = possibilities[choice]
    return distance[n][m], dels


def in_cost(in_char, vowls_in):
    if (in_char in vocals) and not(vowls_in):
        return 0.3
    else:
        return 1


def sub_cost(char1, char2, word_embedding):
    vek1 = word_embedding[ord(char1)]
    vek2 = word_embedding[ord(char2)]
    return min((1 - vek1@vek2)*2,1)


def del_cost(del_char, table, freelo,freelo_amount, scaler=0.75):
    scal_cost = scaler**(table[del_char]-1)
    return min(freelo_amount,scal_cost) if freelo else scal_cost


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


def get_word_dic_distance(word, dic, word_embedding, sort=True, progress=True, orig_word=None, cheap_deletions=True):
    if orig_word is None:
        orig_word = word
    char_app = char_apparence(orig_word)
    vowls_in = vowl_checker(orig_word)
    if len(word)>20:  # Filter out links and other uncomprehensable stuff
        distance = [(sample_word,1,0) for sample_word in dic]
    else:
        distance = []
        for sample_word in (tqdm(dic) if progress else dic):
            distance.append((sample_word, *levenshteinDistance(sample_word, word, word_embedding,char_app,vowls_in,cheap_deletions)))
    if sort:
        distance.sort(key=itemgetter(1))
    words, values, dels = zip(*distance)
    values = np.array(values)
    max_value = np.max(values)
    values = max_value - values
    values = util.softmax(np.array(values),theta=4)
    distance = list(map(list,zip(words, values, dels)))
    return distance


if __name__ == '__main__':
    letter_begin = util.load_dictionary("DATA/dictionaries/bert_letter_begin.txt")
    number_begin = util.load_dictionary("DATA/dictionaries/bert_number_begin.txt")
    dic = letter_begin+number_begin
    word_embedding = util.load_pickle("visual_embeddings.pkl")
    distance = get_word_dic_distance("ɋ˯İՅ", dic, word_embedding, cheap_deletions=False)
    for i in range(40):
        print(distance[i])