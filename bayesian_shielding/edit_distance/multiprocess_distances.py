import sys
sys.path.append("..")
from edit_distance.substring_distance import Sub_dist
import multiprocessing
from functools import reduce
from util.utility import mylower,get_full_word_dict
import numpy as np

full_word_dic = get_full_word_dict
def distance_words(tasks:list,lev_handler:Sub_dist):
    results = []
    for task in tasks:
        word = task[0]
        probs = lev_handler.get_sentence_hypothesis(sentence)
        results.append([probs,*task[1:]])
    return results

def multiprocess_word_distances(sentences:list,lev_handler:Sub_dist,transpo_dict = {}):
    cpu_count = multiprocessing.cpu_count()
    work_words = []
    preresults = []
    for a,sentence in enumerate(sentences):
        for b,word in enumerate(sentence):
            useword = mylower(word)
            if useword in transpo_dict:
                preresults.append([transpo_dict[useword],a,b])
            else:
                work_words.append([useword,a])
    splitl = len(work_words)/(cpu_count-1)
    split_words = []
    for i in range(cpu_count-1):
        split_words.append([work_words[int(round(i*splitl)):int(round((i+1)*splitl))],lev_handler])
    with multiprocessing.Pool(processes=cpu_count-1) as pool:
        results = pool.starmap(distance_words,split_words)
    for i,part in enumerate(results):
        for j,wordpart in enumerate(part):
            word = split_words[i][0][j][0]
            transpo_dict[word] = wordpart[0]
    results = list(reduce(lambda x,y:x+y,results))
    out = [[[] for sentence in line] for line in dataset]
    for result in results:
        out[result[1]][result[2]].append(result[0])
    for preres in preresults:
        out[preres[1]][preres[2]].insert(preres[3],preres[0])
    return out