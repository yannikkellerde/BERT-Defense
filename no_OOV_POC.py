import numpy as np
import util
from tqdm import tqdm,trange
from edit_distance import get_word_dic_distance
from operator import itemgetter
from letter_stuff import trenner_punctuations
import sys

letter_dic = util.load_dictionary("DATA/dictionaries/bert_letter_begin.txt")
number_dic = util.load_dictionary("DATA/dictionaries/bert_number_begin.txt")
punct_dic = util.load_dictionary("DATA/dictionaries/bert_punctuations.txt")
full_word_dic = letter_dic + number_dic + punct_dic
piece_dict = util.load_dictionary("DATA/dictionaries/bert_morphemes.txt")
double_dic = full_word_dic+piece_dict
word_embedding = util.load_pickle("visual_embeddings.pkl")
freq_dict = util.load_freq_dict()

def cut_care_about(probs,care_abouts_prob):
    for i in range(len(probs)):
        if probs[i][1] < care_abouts_prob:
            return i
    return len(probs)

def priorize(probs):
    dic,vals,dels = zip(*probs)
    my_mass = sum(vals)
    prior = util.get_prior([x[0] for x in dic],freq_dict=freq_dict)
    vals = np.array(vals)*prior
    vals = vals*(my_mass/sum(vals))
    out = [list(x) for x in zip(dic,vals,dels)]
    return out

def word_piece_distance(word,care_abouts_prob = 0.01,inner_amount=5):
    word=util.mylower(word)
    probs = get_word_dic_distance(word,full_word_dic,word_embedding,True,False)
    probs = [[[x[0]]]+list(x[1:]) for x in probs]
    probs = priorize(probs)
    probs.sort(key=itemgetter(1),reverse=True)
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
        left_probs = get_word_dic_distance(word[(len(word)-left):],my_dict,word_embedding,True,False,orig_word=word)
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
                    if left==1:
                        print(rein)
                    prob_left_dict[rein[2]].append((rein[0],old[1]*(rein[1]/insum)))
                else:
                    probs.append([rein[0],old[1]*(rein[1]/insum),rein[2]])
    probs.sort(key=itemgetter(1),reverse=True)
    probsum = sum(x[1] for x in probs)
    for p in probs:
        p[1] /= probsum
    return probs

def check_for_some_text(dataset):
    transpo_dict = {}
    out = []
    for line in tqdm(dataset):
        my_line = []
        out.append(my_line)
        for sentence in line:
            words = []
            my_line.append(words)
            for word in sentence:
                print(word)
                if word in transpo_dict:
                    problies = transpo_dict[word]
                else:
                    problies = word_piece_distance(word)
                    transpo_dict[word] = problies[:100]
                my_word = "".join(problies[0][0])
                print(my_word)
                words.append(my_word)
    return out

if __name__ == '__main__':
    dataset = util.load_and_preprocess_dataset("DATA/test-scoreboard-dataset.txt")
    out_dataset = check_for_some_text(dataset[250:270])
    util.write_dataset("preprocessed.txt",out_dataset)
    #res = word_piece_distance(*sys.argv[1:])
    #res.sort(key=itemgetter(1),reverse=True)
    #print(res[:10])