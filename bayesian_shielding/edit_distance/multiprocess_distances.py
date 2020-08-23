import sys
sys.path.append("..")
from edit_distance.edit_distance import get_word_dic_distance
import multiprocessing
from functools import reduce
from util.util import mylower,get_full_word_dict
import numpy as np

full_word_dic = get_full_word_dict
def distance_words(tasks,word_embedding):
    results = []
    for task in tasks:
        word = task[0]
        probs = get_word_dic_distance(word,full_word_dic,word_embedding,cheap_actions=True,keeporder=True,progress=False)
        results.append([probs,*task[1:]])
    return results

def multiprocess_word_distances(dataset,word_embedding,transpo_dict = {}):
    cpu_count = multiprocessing.cpu_count()
    work_words = []
    preresults = []
    for a,line in enumerate(dataset):
        for b,sentence in enumerate(line):
            for c,word in enumerate(sentence):
                useword = mylower(word)
                if useword in transpo_dict:
                    preresults.append([transpo_dict[useword],a,b,c])
                else:
                    work_words.append([useword,a,b])
    splitl = len(work_words)/(cpu_count-1)
    split_words = []
    for i in range(cpu_count-1):
        split_words.append([work_words[int(round(i*splitl)):int(round((i+1)*splitl))],word_embedding])
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