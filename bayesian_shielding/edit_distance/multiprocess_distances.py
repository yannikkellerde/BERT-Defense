import sys,os
import multiprocessing
from functools import reduce
sys.path.append("..")
from util.utility import split_equal
from edit_distance.substring_distance import Sub_dist

def distance_words(sentences: list, lev_handler: Sub_dist):
    results = []
    for sentence in sentences:
        results.append(lev_handler.get_sentence_hypothesis(sentence))

def multiprocess_prior(lev_handler: Sub_dist, sentences: list):
    processes = multiprocessing.cpu_count()-1
    mult_args = list(zip(split_equal(sentences,processes),[lev_handler]*processes))
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.starmap(distance_words, mult_args)
    return reduce(lambda x, y: x+y, results)