import sys,os
import multiprocessing
from tqdm import tqdm
from functools import reduce
sys.path.append("..")
from util.utility import split_equal
from edit_distance.substring_distance import Sub_dist

def distance_words(sentences: list, lev_handler: Sub_dist):
    results = []
    pid = multiprocessing.current_process().pid
    for sentence in tqdm(sentences):
        with open(f"logs{pid}.txt", "a") as f:
            f.write(sentence)
        print(sentence)
        results.append(lev_handler.get_sentence_hypothesis(sentence))
    return results

def multiprocess_prior(lev_handler: Sub_dist, sentences: list):
    processes = multiprocessing.cpu_count()-1
    mult_args = list(zip(split_equal(sentences,processes),[lev_handler]*processes))
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.starmap(distance_words, mult_args)
    return reduce(lambda x, y: x+y, results)