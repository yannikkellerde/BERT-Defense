import numpy as np
import util
from tqdm import tqdm,trange
from edit_distance import get_word_dic_distance
from operator import itemgetter
from letter_stuff import trenner_punctuations
import sys

full_word_dic = util.load_dictionary("DATA/bert_wiki_full_words.txt")
piece_dict = util.load_dictionary("DATA/bert_wiki_word_pieces.txt")
double_dic = full_word_dic+piece_dict
word_embedding = util.load_pickle("visual_embeddings.pkl")

def cut_care_about(probs,care_abouts_prob):
    for i in range(len(probs)):
        if probs[i][1] < care_abouts_prob:
            return i
    return len(probs)

def word_piece_distance(word,care_abouts_prob = 0.001,inner_amount=5):
    word=word.lower()
    probs = get_word_dic_distance(word,full_word_dic,word_embedding,True,True)
    probs = [[[x[0]]]+list(x[1:]) for x in probs]
    prob_cut = cut_care_about(probs,care_abouts_prob)
    if prob_cut == 0:
        return probs
    max_left = max(x[2] for x in probs[:prob_cut])
    prob_left_dict = {i:list() for i in range(max_left+1)}
    for i in range(prob_cut-1,-1,-1):
        prob = probs[i]
        for j in range(i):
            if probs[i][0][0] in probs[j][0][0]:
                del probs[i]
                break
        else:
            if prob[2]>0:
                prob_left_dict[prob[2]].append(prob[:2])
                del probs[i]
    for left in range(max_left,-1,-1):
        if len(prob_left_dict[left]) == 0:
            continue
        if word[(len(word)-left)-1] in trenner_punctuations:
            my_dict = double_dic
        else:
            my_dict = piece_dict
        left_probs = get_word_dic_distance(word[(len(word)-left):],my_dict,word_embedding,True,True,orig_word=word)
        for i in range(len(left_probs)-1,-1,-1):
            for j in range(i):
                if left_probs[i][0][0] in left_probs[j][0][0]:
                    del left_probs[i]
                    break
        for old in prob_left_dict[left]:
            left_cut = inner_amount//len(old[0])+1
            inner_reins = []
            for new in left_probs[:left_cut]:
                inner_reins.append([old[0]+[new[0]],new[1],new[2]])
            insum = sum([x[1] for x in inner_reins])
            for rein in inner_reins:
                if rein[2]>0:
                    prob_left_dict[rein[2]].append((rein[0],old[1]*(rein[1]/insum)))
                else:
                    probs.append([rein[0],old[1]*(rein[1]/insum),rein[2]])
    probs.sort(key=itemgetter(1),reverse=True)
    probsum = sum(x[1] for x in probs)
    for p in probs:
        p[1] /= probsum
    return probs
if __name__ == '__main__':
    res = word_piece_distance(*sys.argv[1:])
    print(res[:10])