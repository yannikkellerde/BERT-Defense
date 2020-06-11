import numpy as np
import util
from tqdm import tqdm,trange
from edit_distance import get_word_dic_distance

full_word_dic = util.load_dictionary("DATA/bert_wiki_full_words.txt")
piece_dict = util.load_dictionary("DATA/bert_wiki_word_pieces.txt")
dic = full_word_dic+piece_dict
word_embedding = util.load_pickle("visual_embeddings.pkl")

def cut_care_about(probs,care_abouts_prob):
    for i in range(len(probs)):
        if probs[i] < care_abouts_prob:
            return i-1
    return len(probs)

def word_piece_distance(word,care_abouts_prob = 0.05):
    probs = get_word_dic_distance(word,full_word_dic,word_embedding,True,True)
    prob_cut = care_abouts_prob(probs,care_abouts_prob)
    lefts = set(x[2] for x in probs[:prob_cut] if x>1)
    reindamits = []
    for left in lefts:
        left_probs = get_word_dic_distance(word[(len(word)-left):],piece_dict,word_embedding,True,True,orgi_word=word)
        left_cut = care_abouts_prob(left_probs,care_abouts_prob)
        for i,old in enumerate(probs[:prob_cut]):
            if old[2] == left:
                for prob in left_probs[:left_cut]:
                    new_prob = old[1]*prob[1]
                    if new_prob > care_abouts_prob:
                        reindamits.append([[old[0],prob[0]],new_prob,prob[2]])
    

if __name__ == '__main__':
    

