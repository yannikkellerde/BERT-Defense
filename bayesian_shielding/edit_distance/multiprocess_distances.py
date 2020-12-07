import sys,os
import multiprocessing
from tqdm import tqdm
from functools import reduce
import pickle
from shutil import rmtree
sys.path.append("..")
from util.utility import split_equal, preprocess_sentence
from edit_distance.substring_distance import Sub_dist
base_path = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))

def distance_words(sentences: list, lev_handler: Sub_dist,store_path:str or None,pnum:int):
    results = []
    for sentence in tqdm(sentences):
        tokens = preprocess_sentence(sentence)
        results.append(lev_handler.get_sentence_hypothesis(tokens))
    if store_path is not None:
        with open(os.path.join(store_path,f"{pnum}.pkl"),"wb") as f:
            pickle.dump(results,f)
        del results
        return None
    return results

def multiprocess_prior(lev_handler: Sub_dist, sentences: list,store_path=None):
    if store_path is not None:
        if os.path.isdir(store_path):
            rmtree(store_path)
        os.makedirs(store_path)
    processes = multiprocessing.cpu_count()-1
    mult_args = list(zip(split_equal(sentences,processes),[lev_handler]*processes,[store_path]*processes,range(processes)))
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.starmap(distance_words, mult_args)
    if store_path is None:
        return reduce(lambda x, y: x+y, results)