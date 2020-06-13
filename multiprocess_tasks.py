from edit_distance import get_word_dic_distance
import multiprocessing
from functools import reduce

def distance_words(tasks,dictionary,word_embedding):
    results = []
    for task in tasks:
        word = task[0]
        if word.lower() in dictionary:
            tmp = [(x,0) for x in dictionary]
            tmp[dictionary.index(word.lower())] = (word.lower(),1)
            results.append([tmp,*task[1:]])
        else:
            results.append([get_word_dic_distance(word, dictionary, word_embedding, sort=False, progress=False),*task[1:]])
    return results

def multiprocess_word_distances(dataset,dictionary,word_embedding):
    cpu_count = multiprocessing.cpu_count()
    work_words = []
    for a,line in enumerate(dataset):
        for b,sentence in enumerate(line):
            for word in sentence:
                work_words.append([word,a,b])
    splitl = len(work_words)/(cpu_count-1)
    split_words = []
    for i in range(cpu_count-1):
        split_words.append([work_words[int(round(i*splitl)):int(round((i+1)*splitl))],dictionary,word_embedding])
    with multiprocessing.Pool(processes=cpu_count-1) as pool:
        results = pool.starmap(distance_words,split_words)
    results = list(reduce(lambda x,y:x+y,results))
    out = [[[] for sentence in line] for line in dataset]
    for result in results:
        out[result[1]][result[2]].append(result[0])
    return out