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
pseudo_morphemes = ["'t","'s"]

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

def word_piece_distance(word,word_embedding,allow_combo_words=True):
    min_care_abouts_prob = 0.01
    word=util.mylower(word)
    if not allow_combo_words:
        return get_word_dic_distance(word,full_word_dic,word_embedding,False,False,cheap_deletions=False)
    probs = get_word_dic_distance(word,full_word_dic,word_embedding,False,False,cheap_deletions=True)
    if len(word)>20:   # Filter out long links and other incomprehensible stuff
        return (probs,tuple())
    #probs = priorize(probs)
    probs_before_sort = probs[:]
    probs.sort(key=itemgetter(1),reverse=True)
    care_abouts_prob = min_care_abouts_prob
    for p in probs:
        if p[1]<min_care_abouts_prob:
            break
        if p[2]==0:
            care_abouts_prob = p[1]
            break
    inner_amount = care_abouts_prob/2
    prob_cut = cut_care_about(probs,care_abouts_prob)
    if prob_cut == 0:
        return (probs_before_sort,tuple())
    max_left = max(x[2] for x in probs[:prob_cut])
    prob_left_dict = {i:list() for i in range(max_left+1)}
    combostuff = []
    for i in range(prob_cut-1,-1,-1):
        prob = [[probs[i][0]]]+list(probs[i][1:])
        for j in range(i):
            if probs[i][0] in probs[j][0]:
                probs[i][1] = 0
                break
        else:
            if prob[2]>0:
                prob_left_dict[prob[2]].append(prob[:2])
                probs[i][1] = 0
    for left in range(max_left,-1,-1):
        if len(prob_left_dict[left]) == 0:
            continue
        if word[(len(word)-left)-1] in trenner_punctuations:
            my_dict = double_dic+pseudo_morphemes
        else:
            my_dict = piece_dict+pseudo_morphemes
        left_probs = get_word_dic_distance(word[(len(word)-left):],my_dict+pseudo_morphemes,word_embedding,True,False,orig_word=word)
        for i in range(len(left_probs)-1,-1,-1):
            for j in range(i):
                if left_probs[i][0][0] in left_probs[j][0][0]:
                    del left_probs[i]
                    break
        for old in prob_left_dict[left]:
            inner_reins = []
            probsum = 0
            for new in left_probs:
                probsum += new[1]
                if old[1]*(new[1]/probsum)<inner_amount:
                    probsum -= new[1]
                    break
                if new[0] in pseudo_morphemes:
                    putin = list(new[0])
                else:
                    putin = [new[0]]
                if new[0] in piece_dict or new[0] in pseudo_morphemes:
                    putin = ["##"+x for x in putin]
                inner_reins.append([old[0]+putin,new[1],new[2]])
            for rein in inner_reins:
                if rein[2]>0:
                    if rein[2]>=left:
                        continue
                    prob_left_dict[rein[2]].append((rein[0],old[1]*(rein[1]/probsum)))
                else:
                    combostuff.append((rein[0],old[1]*(rein[1]/probsum)))
    combostuff.sort(key=itemgetter(1),reverse=True)
    probsum = sum(x[1] for x in probs)
    for p in probs_before_sort:
        p[1] /= probsum
    return (probs_before_sort,tuple(combostuff))

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
    #dataset = util.load_and_preprocess_dataset("DATA/test-scoreboard-dataset.txt")
    #out_dataset = check_for_some_text(dataset[250:270])
    #util.write_dataset("preprocessed.txt",out_dataset)
    word_embedding = util.load_pickle("visual_embeddings.pkl")
    freq_dict = util.load_freq_dict()
    res,combostuff = word_piece_distance(*sys.argv[1:],word_embedding)
    res.sort(key=itemgetter(1),reverse=True)
    print(res[:50],combostuff[:50])