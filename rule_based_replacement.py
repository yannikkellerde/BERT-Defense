"""Undo the adversarial attacks on a dataset."""
import numpy as np
import sys
from util import cosine_similarity,write_dataset,load_unlabeled_dataset,load_pickle
from tqdm import tqdm
import math
import time
from letter_stuff import *

def calculate_distance(word_a,word_b):
    pass

def get_closest(vec_dict,test_chars,target_vec):
    """Find the visually closest charcter in test_chars to the target vector target_vec
    using the visual character embeddings from vec_dict"""

    max_sim = -1
    best_char = ""
    for char in test_chars:
        sim = cosine_similarity(vec_dict[ord(char)],target_vec)
        if sim >= max_sim:
            max_sim = sim
            best_char = char
    return best_char, max_sim

def get_closeness_list(vec_dict,test_chars,target_vec):
    """Sort the characters in test_chars by their visual distance to target_vec 
    using the embeddings from vec_dict and return them together with their distance."""

    unsorted = []
    for char in test_chars:
        unsorted.append((cosine_similarity(vec_dict[ord(char)],target_vec),char))
    return list(sorted(unsorted,reverse=True))

def fuzz_word(word,vec_dict):
    """A generator that generates letter sequences out of normal characters in order
    of their closeness to the given word."""

    def hash_list(a_list):
        my_hash = 0
        for i,num in enumerate(a_list):
            my_hash+=num*(max_num**i)
    def unhash_list(a_hash):
        my_list = []
        for i in range(word_len):
            if i<(word_len-1):
                my_list.append((a_hash%(max_num**(i+1)))//(max_num**i))
            else:
                my_list.append((a_hash//(max_num**i)))
        return my_list
    def add_new_endpoints(endpoints,already_visited,curr_pos):
        for i in range(word_len):
            if curr_pos+max_num**i not in already_visited:
                if i<(word_len-1):
                    if ((curr_pos%(max_num**(i+1)))//(max_num**i))<word_len-1:
                        endpoints.add(curr_pos+max_num**i)
                else:
                    if (curr_pos//(max_num**i))<(word_len-1):
                        endpoints.add(curr_pos+max_num**i)
    max_num = min(len(all_chars),int((2**28)**(1/len(word))))
    word_len = len(word)
    curr_pos = 0
    closeness_lists = [get_closeness_list(vec_dict,all_chars,vec_dict[ord(x)]) for x in word]
    visit_poses = set((curr_pos,))
    endpoints = set()
    add_new_endpoints(endpoints,visit_poses,curr_pos)
    while len(endpoints)>0:
        out = ""
        for i,p in enumerate(unhash_list(curr_pos)):
            out+=closeness_lists[i][p][1]
        yield out
        best_sim = -np.inf
        for endpoint in endpoints:
            ep_list = unhash_list(endpoint)
            sim = 0
            for i,p in enumerate(ep_list):
                sim+=closeness_lists[i][p][0]
            if sim >= best_sim:
                curr_pos = endpoint
                best_sim = sim
        endpoints.discard(curr_pos)
        visit_poses.add(curr_pos)
        add_new_endpoints(endpoints,visit_poses,curr_pos)

def get_replaced_dataset(dataset,vec_dict):
    """Replaces the characters in a whole dataset with their visually closest
    normal letters and returns it"""

    out_dataset = []
    for pair in tqdm(dataset):
        pair_ray = []
        out_dataset.append(pair_ray)
        for sentence in pair:
            sent_str = ""
            for letter in sentence:
                if letter in letters or letter in punctuations:
                    sent_str += letter
                elif ord(letter) in vec_dict:
                    sent_str += get_closest(vec_dict,letters,vec_dict[ord(letter)])[0]
            pair_ray.append(sent_str[0]+sent_str[1:].lower())
    return out_dataset

if __name__ == '__main__':
    #write_dataset("replaced_data.txt",get_replaced_dataset(load_unlabeled_dataset("DATA/test-hex06-dataset.txt"),
    #              load_pickle("visual_embeddings.pkl")))
    vec_dict = load_pickle("visual_embeddings.pkl")
    gen = fuzz_word(sys.argv[1],vec_dict)
    for trial in gen:
        print(trial)
        if trial=="black":
            break