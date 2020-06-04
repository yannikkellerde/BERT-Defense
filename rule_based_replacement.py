import numpy as np
import sys
from util import cosine_similarity,write_dataset,load_unlabeled_dataset,load_embedding_file
from tqdm import tqdm
import math
import time

normal_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
keep_letters = ",.?-"

def get_closest(vec_dict,test_chars,target_vec):
    max_sim = -1
    best_char = ""
    for char in test_chars:
        sim = cosine_similarity(vec_dict[ord(char)],target_vec)
        if sim >= max_sim:
            max_sim = sim
            best_char = char
    return best_char, max_sim

def get_closeness_list(vec_dict,test_chars,target_vec):
    unsorted = []
    for char in test_chars:
        unsorted.append((cosine_similarity(vec_dict[ord(char)],target_vec),char))
    return list(sorted(unsorted,reverse=True))

def fuzz_word(word,vec_dict):
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
    my_normal = normal_letters+keep_letters
    max_num = min(len(my_normal),int((2**28)**(1/len(word))))
    word_len = len(word)
    curr_pos = 0
    closeness_lists = [get_closeness_list(vec_dict,my_normal,vec_dict[ord(x)]) for x in word]
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
    out_dataset = []
    for pair in tqdm(dataset):
        pair_ray = []
        out_dataset.append(pair_ray)
        for sentence in pair:
            sent_str = ""
            for letter in sentence:
                if letter in normal_letters or letter in keep_letters:
                    sent_str += letter
                elif ord(letter) in vec_dict:
                    sent_str += get_closest(vec_dict,normal_letters,vec_dict[ord(letter)])[0]
            pair_ray.append(sent_str[0]+sent_str[1:].lower())
    return out_dataset

if __name__ == '__main__':
    #write_dataset("replaced_data.txt",get_replaced_dataset(load_unlabeled_dataset("DATA/test-hex06-dataset.txt"),
    #              load_embedding_file("visual_embeddings.pkl")))
    vec_dict = load_embedding_file("visual_embeddings.pkl")
    gen = fuzz_word(sys.argv[1],vec_dict)
    for trial in gen:
        print(trial)
        if trial=="kitchen":
            break