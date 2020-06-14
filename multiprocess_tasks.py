from edit_distance import get_word_dic_distance
import multiprocessing
from functools import reduce
from util import mylower
import numpy as np

def distance_words(tasks,dictionary,word_embedding):
    results = []
    for task in tasks:
        word = task[0]
        probs_mit_wort = get_word_dic_distance(word, dictionary, word_embedding, sort=False, progress=False)
        probs_ohne_wort = np.array([x[1] for x in probs_mit_wort])
        results.append([probs_ohne_wort,*task[1:]])
    return results

def multiprocess_word_distances(dataset,dictionary,word_embedding,transpo_dict = {}):
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
        split_words.append([work_words[int(round(i*splitl)):int(round((i+1)*splitl))],dictionary,word_embedding])
    with multiprocessing.Pool(processes=cpu_count-1) as pool:
        results = pool.starmap(distance_words,split_words)
    for i,part in enumerate(results):
        for j,wordpart in enumerate(part):
            transpo_dict[split_words[i][0][j][0]] = wordpart[0]
    results = list(reduce(lambda x,y:x+y,results))
    out = [[[] for sentence in line] for line in dataset]
    for result in results:
        out[result[1]][result[2]].append(result[0])
    for preres in preresults:
        out[preres[1]][preres[2]].insert(preres[3],preres[0])
    return np.array(out)