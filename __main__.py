import logging
# create logger with 'root'
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('__main__.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(file_formatter)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(console_formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

import numpy as np
from bert_posterior import bert_posterior, format_dict
from edit_distance import get_word_dic_distance
from util import load_dictionary,load_pickle,load_and_preprocess_dataset
from letter_stuff import sentence_ends
from util import multiprocess_word_distances
import time

if __name__ == '__main__':
    logger.info("loading dictionary, word embedding and dataset")
    dictionary = load_dictionary("DATA/bert_wiki_full_words.txt")
    word_embedding = load_pickle("visual_embeddings.pkl")
    bert_dict = format_dict(dictionary)
    dataset = load_and_preprocess_dataset("DATA/test-scoreboard-dataset.txt")
    in_data = dataset[0:1]
    logger.info(in_data)
    logger.info("calculating distances")
    start = time.perf_counter()
    priors = multiprocess_word_distances(in_data,dictionary,word_embedding)
    posterior_sentences = [[[] for _sentence in line] for line in priors]
    for l,line in enumerate(priors):
        for s,sentence_prior in enumerate(line):
            prior_sentence = [list(sorted([(y[1],y[0]) for y in x],reverse=True))[0][1] for x in sentence_prior]
            logger.info("Only distance normalized sentence: "+str(prior_sentence))
            if prior_sentence[-1] not in sentence_ends:
                tmp = [(x,1/len(sentence_ends) if x in sentence_ends else 0) for x in dictionary]
                sentence_prior.append(tmp)
            bert_prior = np.array([[x[1] for x in p] for p in sentence_prior])
            logger.info("calculating posterior")
            posterior = bert_posterior(bert_prior,bert_dict,10)
            out_sentence = [dictionary[np.argmax(p)] for p in posterior]
            logger.info("Posterior sentence"+str(out_sentence))
            posterior_sentences[l][s] = out_sentence

    logger.info(f"time taken: {time.perf_counter()-start}")

    logger.info("full posterior:")
    logger.info(posterior_sentences)